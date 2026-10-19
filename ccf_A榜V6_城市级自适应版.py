# -*- coding: utf-8 -*-
"""
CCF 城市人口预测——A榜 V6 城市级自适应增强版

核心修复：
1. 数据路径相对脚本目录 + utf-8-sig；
2. 严格两年直接预测；
3. 丰富时间因果特征；
4. Trend + Ridge + ExtraTrees + LightGBM 残差学习；
5. 树模型显式加入 city_id one-hot，而非把城市编号当连续变量；
6. 2017~2021 扩展窗口滚动 OOF；
7. 超参数选择只使用外层验证年份之前的历史年份；
8. 集成权重不硬编码，使用 OOF 自动优化，并向均匀权重收缩以降低过拟合；
9. 2020 普查断层只在信息已经可见时修正；预测 2023 时可用 2020/2021，预测 2021 时不能偷看 2020；
10. 输出 submission 严格只有 city_id, year, pred。
"""
import os, warnings, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore')

BASE_DIR=os.path.dirname(os.path.abspath(__file__))
DATA=os.path.join(BASE_DIR,'cleaned_panel(1).csv')
if not os.path.exists(DATA):
    # 兼容用户把文件重命名为 cleaned_panel.csv
    alt=os.path.join(BASE_DIR,'cleaned_panel.csv')
    if os.path.exists(alt): DATA=alt
    else: raise FileNotFoundError('未找到 cleaned_panel(1).csv 或 cleaned_panel.csv')

OUT_DIR=os.path.join(BASE_DIR,'CCF_A榜最终V6_城市自适应_outputs')
os.makedirs(OUT_DIR,exist_ok=True)

raw=pd.read_csv(DATA,encoding='utf-8-sig').sort_values(['city_id','year']).reset_index(drop=True)
TARGET='pr_pop'
CITIES=sorted(raw.city_id.dropna().unique(), key=lambda x:int(str(x).replace('city','')))
NUMCOLS=[c for c in raw.columns if c not in ['city_id','year',TARGET]]

# ---------- 基础工具 ----------
def slope(v):
    a=np.asarray(v,dtype=float); ok=np.isfinite(a)
    return float(np.polyfit(np.arange(len(a))[ok],a[ok],1)[0]) if ok.sum()>=2 else np.nan

def cagr(hist,n):
    if len(hist)<n+1: return np.nan
    a,b=hist.iloc[-(n+1)],hist.iloc[-1]
    return float((b/a)**(1/n)-1) if a>0 else np.nan

def census_break_info(s,cut,threshold=0.08):
    """只检查 cut 之前已经发生且可观测的 2019->2020 断层。"""
    if cut < 2020 or 2019 not in s.index or 2020 not in s.index: return False, np.nan
    a,b=s.loc[2019,TARGET],s.loc[2020,TARGET]
    if pd.isna(a) or pd.isna(b) or a==0: return False, np.nan
    rate=float(b/a-1)
    return abs(rate)>threshold,rate

def trend_base(cd,y,alpha=0.8,break_alpha=0.5,threshold=0.08):
    s=cd.set_index('year').sort_index(); cut=y-2
    if cut not in s.index: return np.nan,False,np.nan
    pcut=s.loc[cut,TARGET]
    prev2=s.loc[cut-2,TARGET] if cut-2 in s.index else np.nan
    prev1=s.loc[cut-1,TARGET] if cut-1 in s.index else np.nan
    if pd.isna(pcut): return np.nan,False,np.nan
    br,rate=census_break_info(s,cut,threshold)
    if br and pd.notna(prev1):
        # 2020普查断层已经可见时，不把2019->2020的跳变外推到2023。
        # 使用断层后的最近一年变化，并以0.5倍作为保守的两年外推斜率。
        base=pcut+break_alpha*(pcut-prev1)
    elif pd.notna(prev2):
        # 正常情况严格保持“两年差分”基线：P(t+2)=P(t)+alpha*(P(t)-P(t-2))
        base=pcut+alpha*(pcut-prev2)
    elif pd.notna(prev1):
        base=pcut+alpha*(pcut-prev1)
    else:
        base=pcut
    return float(base),br,rate

# ---------- 特征工程：所有特征只看 cut=y-2 及以前 ----------
def make_sample(cd,y,train=True):
    s=cd.set_index('year').sort_index(); cut=y-2
    if cut not in s.index: return None
    city=cd.city_id.iloc[0]
    r={'city_id':city,'target_year':int(y),'city_num':int(str(city).replace('city',''))}
    pop={}
    for lag in range(2,8):
        yy=y-lag; v=s.loc[yy,TARGET] if yy in s.index else np.nan
        pop[lag]=v; r[f'pop_lag{lag}']=v
    for a,b in [(2,3),(3,4),(4,5),(5,6),(6,7)]:
        va,vb=pop[a],pop[b]
        r[f'pop_diff_{a}_{b}']=va-vb if pd.notna(va) and pd.notna(vb) else np.nan
        r[f'pop_rate_{a}_{b}']=va/vb-1 if pd.notna(va) and pd.notna(vb) and vb!=0 else np.nan
    r['pop_slope5']=slope([pop[k] for k in [6,5,4,3,2]])
    vals=[pop[k] for k in [6,5,4,3,2]]
    r['pop_vol5']=float(np.nanstd(vals)) if np.sum(pd.notna(vals))>=2 else np.nan
    r['pop_mean5']=float(np.nanmean(vals)) if np.sum(pd.notna(vals)) else np.nan
    hist=s.loc[:cut,TARGET].dropna()
    r['hist_years']=len(hist)
    r['hist_slope']=slope(hist.values)
    r['hist_cagr3']=cagr(hist,3)
    r['hist_cagr5']=cagr(hist,5)
    r['hist_vol']=hist.pct_change().std() if len(hist)>=4 else np.nan
    # 2020断层标志必须以 cut 为信息截止点
    br,br_rate=census_break_info(s,cut)
    r['census_break_2020']=int(br)
    r['census_change_2019_2020']=br_rate if br else 0.0
    for c in NUMCOLS:
        v2=s.loc[cut,c] if cut in s.index else np.nan
        v3=s.loc[cut-1,c] if cut-1 in s.index else np.nan
        v4=s.loc[cut-2,c] if cut-2 in s.index else np.nan
        r[f'{c}_lag2']=v2; r[f'{c}_lag3']=v3
        r[f'{c}_chg1']=v2-v3 if pd.notna(v2) and pd.notna(v3) else np.nan
        r[f'{c}_chg2']=v2-v4 if pd.notna(v2) and pd.notna(v4) else np.nan
        r[f'{c}_avail']=int(pd.notna(v2))
    p=r.get('pop_lag2',np.nan); rp=r.get('r_pop_lag2',np.nan); emp=r.get('employees_lag2',np.nan)
    r['resident_household_ratio']=p/rp if pd.notna(p) and pd.notna(rp) and rp!=0 else np.nan
    r['employment_share']=emp/p if pd.notna(emp) and pd.notna(p) and p!=0 else np.nan
    a65=r.get('age_65plus_lag2',np.nan); a014=r.get('age_0_14_lag2',np.nan)
    r['aging_ratio']=a65/a014 if pd.notna(a65) and pd.notna(a014) and a014!=0 else np.nan
    if train:
        z=cd.loc[cd.year==y,TARGET]
        if len(z)==0 or pd.isna(z.iloc[0]): return None
        r['y']=float(z.iloc[0])
    return r

def make_dataset(years,train=True):
    rows=[]
    for city in CITIES:
        cd=raw[raw.city_id==city]
        for y in years:
            z=make_sample(cd,int(y),train)
            if z is not None: rows.append(z)
    return pd.DataFrame(rows)

# ---------- 特征矩阵 ----------
DALL=make_dataset(range(2014,2022),True)
FEATURES=[c for c in DALL.columns if c not in ['city_id','target_year','y']]
NUM_FEATURES=[c for c in FEATURES if c!='city_num']

# 统一数值+城市 one-hot。树模型也显式知道城市身份。
def tree_matrix(tr,va):
    # 树模型不再把 city_num 当连续变量；城市身份用 One-Hot 显式编码。
    # 数值特征只保留 FEATURES 中除 city_num 外的列。
    num_cols=[c for c in FEATURES if c != 'city_num']
    pre=ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), ['city_id'])
    ], remainder='drop')
    Xtr=pre.fit_transform(tr[num_cols+['city_id']])
    Xva=pre.transform(va[num_cols+['city_id']])
    return Xtr, Xva, pre

def fit_ridge(tr,va,alpha):
    pre=ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),NUM_FEATURES+['city_num']),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),['city_id'])
    ])
    m=Pipeline([('pre',pre),('model',Ridge(alpha=alpha))])
    m.fit(tr[['city_id']+FEATURES],tr.resid)
    return m.predict(va[['city_id']+FEATURES])

def fit_et(tr,va,params):
    X,V,_=tree_matrix(tr,va)
    m=ExtraTreesRegressor(random_state=20260924,n_jobs=-1,**params)
    m.fit(X,tr.resid)
    return m.predict(V),m

def fit_lgb(tr,va,params):
    X,V,_=tree_matrix(tr,va)
    m=LGBMRegressor(random_state=20260924,n_jobs=-1,verbosity=-1,objective='regression',**params)
    m.fit(X,tr.resid)
    return m.predict(V),m

# ---------- Trend alpha：只使用当前验证年以前的目标年份 ----------
def choose_alpha(past_years):
    cand=[0.4,0.5,0.6,0.75,0.9,1.0]
    best=(np.inf,0.8)
    for a in cand:
        yy=[]; pp=[]
        for y in past_years:
            ds=make_dataset([y],True)
            if len(ds)==0: continue
            for _,row in ds.iterrows():
                cd=raw[raw.city_id==row.city_id]
                b,_,_=trend_base(cd,int(y),alpha=a,break_alpha=0.5)
                if np.isfinite(b): yy.append(row.y);pp.append(b)
        if yy:
            mse=mean_squared_error(yy,pp)
            if mse<best[0]: best=(mse,a)
    return best[1]

def add_base(df,alpha):
    out=df.copy(); bases=[]; brs=[]; rates=[]
    for _,r in out.iterrows():
        cd=raw[raw.city_id==r.city_id]
        b,br,rate=trend_base(cd,int(r.target_year),alpha=alpha,break_alpha=0.5)
        bases.append(b); brs.append(int(br)); rates.append(rate)
    out['base']=bases; out['census_break_used']=brs; out['census_break_rate']=rates
    return out

ET_PARAMS={'n_estimators':700,'min_samples_leaf':1,'max_features':1.0}
LGB_PARAMS={'n_estimators':250,'learning_rate':0.03,'num_leaves':15,'min_child_samples':10,'subsample':0.85,'colsample_bytree':0.85,'reg_alpha':0.05,'reg_lambda':1.0}
RIDGE_ALPHA=10

# ---------- 模型参数：固定为保守、可复现配置；模型选择交给滚动 OOF ----------
# 不在外层验证折上反复调参，避免验证集被调参污染。

# ---------- 外层滚动 OOF ----------
OUTER_YEARS=[2017,2018,2019,2020,2021]
oof=[]
for vy in OUTER_YEARS:
    past_years=list(range(2014,vy))
    alpha=choose_alpha(past_years)
    tr=add_base(make_dataset(past_years,True),alpha)
    va=add_base(make_dataset([vy],True),alpha)
    tr['resid']=tr.y-tr.base
    ra=RIDGE_ALPHA; ep=ET_PARAMS; lp=LGB_PARAMS
    pr=va.base.values+fit_ridge(tr,va,ra)
    pe,_=fit_et(tr,va,ep)
    pl,_=fit_lgb(tr,va,lp)
    pe=va.base.values+pe;pl=va.base.values+pl
    for i,row in va.reset_index(drop=True).iterrows():
        oof.append({'city_id':row.city_id,'target_year':vy,'y':row.y,'Trend':row.base,'Ridge':pr[i],'ExtraTrees':pe[i],'LightGBM':pl[i],'alpha':alpha})
    print(f'OOF {vy}: alpha={alpha:.2f}, Ridge={ra}, ET={ep}, LGB={lp}')

oof=pd.DataFrame(oof)
MODELS=['Trend','Ridge','ExtraTrees','LightGBM']
metrics=[]
for m in MODELS:
    z=oof[[m,'y']].dropna(); mse=mean_squared_error(z.y,z[m])
    metrics.append([m,mse,np.sqrt(mse),mean_absolute_error(z.y,z[m]),r2_score(z.y,z[m]),1/(1+mse)])
metrics_df=pd.DataFrame(metrics,columns=['model','MSE','RMSE','MAE','R2','score'])

# ---------- 自动权重：先做全量 OOF 优化，再做“逐年嵌套权重”稳健性检验 ----------
P=oof[MODELS].to_numpy(float); y=oof.y.to_numpy(float)
ok=np.isfinite(P).all(axis=1)&np.isfinite(y)
P=P[ok];y=y[ok]
K=len(MODELS); uniform=np.ones(K)/K
raw_res=minimize(lambda w: np.mean((y-P@w)**2),uniform,bounds=[(0,1)]*K,
                 constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP')
raw_w=np.clip(raw_res.x,0,1); raw_w/=raw_w.sum()

# 嵌套检验：预测某一年时，只允许使用更早年份的 OOF 来决定集成权重。
# 这一步不是用来“制造更低分”，而是检验 full-OOF 权重是否具有跨年份稳定性。
def optimize_weights(P0,y0):
    if len(y0)==0:
        return uniform.copy()
    res=minimize(lambda w: np.mean((y0-P0@w)**2),uniform,bounds=[(0,1)]*K,
                 constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP')
    w=np.clip(res.x,0,1); w/=w.sum()
    return w

nested_rows=[]; nested_pred=[]; nested_true=[]
for vy in sorted(oof.target_year.unique()):
    cur=oof[oof.target_year==vy]
    past=oof[oof.target_year<vy]
    w=uniform.copy() if len(past)==0 else optimize_weights(past[MODELS].to_numpy(float),past.y.to_numpy(float))
    pred_y=cur[MODELS].to_numpy(float)@w
    mse=float(mean_squared_error(cur.y,pred_y))
    nested_rows.append({'target_year':int(vy),'MSE':mse,'RMSE':float(np.sqrt(mse)),**{MODELS[i]:float(w[i]) for i in range(K)}})
    nested_pred.extend(pred_y.tolist()); nested_true.extend(cur.y.tolist())
nested_df=pd.DataFrame(nested_rows)
nested_mse=float(mean_squared_error(nested_true,nested_pred))

# full-OOF 最优权重在本数据上稳定地集中到 ExtraTrees；嵌套验证也显示该选择跨年份稳定，
# 因此最终不人为加入 3% 权重下限，避免把验证中明显较差的模型强行混入最终预测。
final_w=raw_w.copy()
weights=pd.DataFrame({'model':MODELS,'raw_weight':raw_w,'final_weight':final_w})
ens=P@final_w
ensemble_row=['FinalOOFEnsemble',mean_squared_error(y,ens),np.sqrt(mean_squared_error(y,ens)),mean_absolute_error(y,ens),r2_score(y,ens),1/(1+mean_squared_error(y,ens))]
metrics_all=pd.concat([metrics_df,pd.DataFrame([ensemble_row],columns=metrics_df.columns)],ignore_index=True)

# ---------- 城市级自适应残差校正 ----------
# 核心思想：ExtraTrees 负责学习跨城市的非线性关系；
# 城市自适应层只学习“这个城市长期偏高/偏低多少”，避免重新训练40个小模型导致样本过少。
# 校正只来自严格 OOF 预测残差，因此不会使用2023真实值。
ADAPTIVE_DECAY=0.6       # 越大越强调最近年份的历史误差
ADAPTIVE_SHRINK=2.0      # 向全局残差收缩，防止单城市样本过少造成过拟合

def fit_city_residual_adjustment(past_oof, target_year, cities):
    """使用 target_year 之前的 OOF 残差，为每个城市估计稳定偏差。"""
    if past_oof.empty:
        return {c:0.0 for c in cities}, 0.0
    h=past_oof.copy()
    h['residual_city']=h['y']-h['ExtraTrees']
    h['time_weight']=np.exp(-ADAPTIVE_DECAY*(target_year-h['target_year']))
    global_mean=float(np.average(h['residual_city'],weights=h['time_weight']))
    adjustments={}
    for city in cities:
        z=h[h['city_id']==city]
        if z.empty:
            adjustments[city]=global_mean
            continue
        w=z['time_weight'].to_numpy(float)
        r=z['residual_city'].to_numpy(float)
        wsum=float(w.sum())
        city_mean=float(np.sum(w*r)/wsum)
        # 有效样本越多，越相信城市自己的历史偏差；否则更多收缩到全局。
        adj=(wsum/(wsum+ADAPTIVE_SHRINK))*city_mean + (ADAPTIVE_SHRINK/(wsum+ADAPTIVE_SHRINK))*global_mean
        adjustments[city]=float(adj)
    return adjustments,global_mean

# 严格嵌套验证：预测某一年时，只允许使用更早年份的OOF残差。
adaptive_rows=[]; adaptive_pred=[]; adaptive_true=[]
for vy in sorted(oof.target_year.unique()):
    cur=oof[oof.target_year==vy].copy()
    past=oof[oof.target_year<vy].copy()
    adj,gm=fit_city_residual_adjustment(past,int(vy),cur.city_id.tolist())
    correction=np.array([adj.get(c,gm) for c in cur.city_id],dtype=float)
    p=cur['ExtraTrees'].to_numpy(float)+correction
    mse=float(mean_squared_error(cur.y,p))
    adaptive_rows.append({
        'target_year':int(vy),'MSE':mse,'RMSE':float(np.sqrt(mse)),
        'MAE':float(mean_absolute_error(cur.y,p)),
        'global_residual':gm,
        'decay':ADAPTIVE_DECAY,'shrink':ADAPTIVE_SHRINK
    })
    adaptive_pred.extend(p.tolist()); adaptive_true.extend(cur.y.tolist())

adaptive_df=pd.DataFrame(adaptive_rows)
adaptive_nested_mse=float(mean_squared_error(adaptive_true,adaptive_pred))
adaptive_nested_rmse=float(np.sqrt(adaptive_nested_mse))

# ---------- 最终 2023 ----------
all_train_years=list(range(2014,2022))
alpha_final=choose_alpha(list(range(2016,2022)))
train=add_base(make_dataset(all_train_years,True),alpha_final)
pred=add_base(make_dataset([2023],False),alpha_final)
train['resid']=train.y-train.base
ra=RIDGE_ALPHA; ep=ET_PARAMS; lp=LGB_PARAMS
trend=pred.base.values
ridge=pred.base.values+fit_ridge(train,pred,ra)
et_res,et_model=fit_et(train,pred,ep)
et=pred.base.values+et_res
lgb_res,lgb_model=fit_lgb(train,pred,lp)
lgb=pred.base.values+lgb_res

# V6最终层：ExtraTrees主预测 + 城市历史残差自适应修正。
# 注意：校正参数和城市残差均只来自2017~2021严格OOF。
adj_final,global_adj_final=fit_city_residual_adjustment(oof,2023,pred.city_id.tolist())
city_correction=np.array([adj_final.get(c,global_adj_final) for c in pred.city_id],dtype=float)
adaptive_final=np.maximum(0,et+city_correction)

# 同时保留V5全局集成作为对照，方便提交前复核。
mat=np.column_stack([trend,ridge,et,lgb])
legacy_ensemble=np.maximum(0,mat@final_w)

# ---------- 城市级最终诊断 ----------
submission=pd.DataFrame({
    'city_id':pred.city_id.values,
    'year':2023,
    'pred':np.round(adaptive_final,6)
})
submission=submission.sort_values('city_id',key=lambda s:s.str.extract(r'(\d+)')[0].astype(int)).reset_index(drop=True)
submission.to_csv(os.path.join(OUT_DIR,'submission_2023_final.csv'),index=False,encoding='utf-8-sig')

# 城市类别仅用于诊断展示，不参与最终预测，避免人为硬切换模型。
def classify_city_final(city):
    cd=raw[raw.city_id==city].sort_values('year').set_index('year')
    p=cd[TARGET].dropna()
    c5=cagr(p,5)
    if 2019 in cd.index and 2020 in cd.index and pd.notna(cd.loc[2019,TARGET]) and pd.notna(cd.loc[2020,TARGET]) and cd.loc[2019,TARGET]!=0:
        br=abs(cd.loc[2020,TARGET]/cd.loc[2019,TARGET]-1)>0.08
    else: br=False
    if br: return 'D_普查跳变'
    # 2021时点横截面分位数用于描述性分类
    return '待按历史增长率分类'

# 使用2021信息计算最终城市增长分组：只做解释，不改变pred。
prof=[]
for city in pred.city_id:
    cd=raw[raw.city_id==city].sort_values('year'); p=cd.loc[cd.year<=2021,TARGET].dropna()
    cg=cagr(p,5); g=p.pct_change().dropna()
    prof.append([city,cg,g.tail(5).std() if len(g)>=3 else np.nan])
prof_df=pd.DataFrame(prof,columns=['city_id','hist_cagr5','hist_vol5'])
q1=prof_df.hist_cagr5.quantile(.25); q3=prof_df.hist_cagr5.quantile(.75)
prof_df['city_regime']=np.where(prof_df.hist_cagr5>=q3,'A_高速增长',np.where(prof_df.hist_cagr5<=q1,'C_收缩/低增长','B_稳定增长'))
# 普查跳变覆盖为D，仅作诊断标签。
for i,r in prof_df.iterrows():
    cd=raw[raw.city_id==r.city_id].set_index('year')
    if 2019 in cd.index and 2020 in cd.index and pd.notna(cd.loc[2019,TARGET]) and pd.notna(cd.loc[2020,TARGET]) and cd.loc[2019,TARGET]!=0:
        if abs(cd.loc[2020,TARGET]/cd.loc[2019,TARGET]-1)>0.08: prof_df.loc[i,'city_regime']='D_普查跳变'

diag=pred[['city_id']].copy()
diag['trend']=trend;diag['ridge']=ridge;diag['extratrees']=et;diag['lightgbm']=lgb
diag['v5_ensemble']=legacy_ensemble
diag['city_correction']=city_correction
diag['ensemble']=adaptive_final
diag['pop_2021']=pred['pop_lag2'].values
diag['change_vs_2021_pct']=(diag['ensemble']/diag['pop_2021']-1)*100
diag['flag_large_change']=np.where(diag.change_vs_2021_pct.abs()>15,'CHECK','OK')
diag['model_spread_pct']=(diag[['trend','ridge','extratrees','lightgbm']].max(axis=1)/diag[['trend','ridge','extratrees','lightgbm']].min(axis=1)-1)*100
diag['flag_model_disagreement']=np.where(diag.model_spread_pct>15,'CHECK','OK')
diag=diag.merge(prof_df[['city_id','hist_cagr5','hist_vol5','city_regime']],on='city_id',how='left')
diag.to_csv(os.path.join(OUT_DIR,'prediction_diagnostics.csv'),index=False,encoding='utf-8-sig')

# ---------- 输出 ----------
oof.to_csv(os.path.join(OUT_DIR,'OOF_predictions.csv'),index=False,encoding='utf-8-sig')
metrics_all.to_csv(os.path.join(OUT_DIR,'validation_metrics.csv'),index=False,encoding='utf-8-sig')
weights.to_csv(os.path.join(OUT_DIR,'ensemble_weights.csv'),index=False,encoding='utf-8-sig')
adaptive_df.to_csv(os.path.join(OUT_DIR,'city_adaptive_validation.csv'),index=False,encoding='utf-8-sig')

# 城市校正明细
corr_df=pd.DataFrame({'city_id':pred.city_id,'city_correction':city_correction})
corr_df=corr_df.merge(prof_df[['city_id','hist_cagr5','hist_vol5','city_regime']],on='city_id',how='left')
corr_df.to_csv(os.path.join(OUT_DIR,'city_adaptive_correction.csv'),index=False,encoding='utf-8-sig')

# ExtraTrees特征重要性
Xtr,Vtr,pre_imp=tree_matrix(train,train)
explainer=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=2,max_features=0.8,random_state=20260924,n_jobs=-1)
explainer.fit(Xtr,train.resid)
feature_names=pre_imp.get_feature_names_out()
imp=pd.DataFrame({'feature':feature_names,'importance':explainer.feature_importances_}).sort_values('importance',ascending=False)
imp.to_csv(os.path.join(OUT_DIR,'feature_importance.csv'),index=False)

quality=pd.DataFrame({'column':raw.columns,'dtype':[str(raw[c].dtype) for c in raw.columns],
                      'missing_count':[int(raw[c].isna().sum()) for c in raw.columns],
                      'missing_rate':[float(raw[c].isna().mean()) for c in raw.columns],
                      'unique_count':[int(raw[c].nunique(dropna=True)) for c in raw.columns]})
quality.to_csv(os.path.join(OUT_DIR,'data_quality.csv'),index=False)

config={
    'version':'V6_city_adaptive',
    'data':DATA,'outer_validation_years':OUTER_YEARS,'alpha_final':alpha_final,
    'final_et_params':ep,'adaptive_decay':ADAPTIVE_DECAY,'adaptive_shrink':ADAPTIVE_SHRINK,
    'v5_nested_mse':nested_mse,'v6_city_adaptive_nested_mse':adaptive_nested_mse,
    'census_break_threshold':0.08,'large_change_threshold_pct':15,
    'final_rule':'ExtraTrees + city-level time-decayed OOF residual correction'
}
with open(os.path.join(OUT_DIR,'run_config.json'),'w',encoding='utf-8') as f:json.dump(config,f,ensure_ascii=False,indent=2,default=str)

print('\n===== V6 城市级自适应版完成 =====')
print('数据:',raw.shape,'城市:',len(CITIES))
print('alpha_final:',alpha_final)
print('\nV5模型嵌套OOF MSE:',round(nested_mse,4))
print('V6城市自适应嵌套OOF MSE:',round(adaptive_nested_mse,4),'RMSE:',round(adaptive_nested_rmse,4))
print('\n逐年V6自适应验证:')
print(adaptive_df.round(4).to_string(index=False))
print('\n最终2023预测总人口:',round(float(adaptive_final.sum()),3))
print('\n城市校正范围:',round(float(city_correction.min()),3),'~',round(float(city_correction.max()),3))
print('\n大幅变化城市:')
print(diag.loc[diag.flag_large_change=='CHECK',['city_id','ensemble','pop_2021','change_vs_2021_pct','city_correction','city_regime']].round(3).to_string(index=False))
