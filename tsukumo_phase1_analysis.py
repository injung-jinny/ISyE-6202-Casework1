from pathlib import Path
import math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    from bokeh.sampledata.us_states import data as US_STATES
except (ImportError, ModuleNotFoundError, RuntimeError):
    US_STATES={}  # Plotting fallback when optional bokeh_sampledata is unavailable

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tsukumo_outputs'; FIG=OUT/'figures'; OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
SEED=62026335; rng=np.random.default_rng(SEED)
season=pd.read_csv(ROOT/'demand_seasonalities.csv')
dist=pd.read_csv(ROOT/'fc_zip3_distance.csv', dtype={'ZIP3':str})
market=pd.read_csv(ROOT/'zip3_market.csv', dtype={'ZIP3':str})
pmf=pd.read_csv(ROOT/'zip3_pmf.csv', dtype={'ZIP3':str})
coord=pd.read_csv(ROOT/'zip3_coordinates.csv', dtype={'ZIP3':str})
msa=pd.read_csv(ROOT/'msa.csv', dtype={'3-digit ZIP code':str})
for d in [dist,market,pmf,coord]: d['ZIP3']=d['ZIP3'].str.zfill(3)
msa['3-digit ZIP code']=msa['3-digit ZIP code'].str.zfill(3)
base=market.merge(coord,on='ZIP3',how='left').merge(pmf,on='ZIP3',how='left').merge(dist,on='ZIP3',how='left')
base['PMF']=base['PMF'].fillna(0.0)
assert abs(base.PMF.sum()-1)<1e-5

CURRENT_MARKET=2_000_000; MARKET_GROWTH=.075; SHARE=.036; SHARE_GROWTH=.20
MARKET_ANNUAL=CURRENT_MARKET*(1+MARKET_GROWTH); TS_ANNUAL=MARKET_ANNUAL*SHARE*(1+SHARE_GROWTH)
PRICE=3000; WEIGHT=60; VOLUME=12; COGS=750; DAYS=364
FC15=['AZ-852','CA-900','CA-945','CO-802','FL-331','GA-303','IL-606','MA-021','MI-481','NC-275','NJ-070','TX-750','TX-770','UT-841','WA-980']
RAD={'GA-303':0,'NY-134':2,'TX-799':3,'AZ-852':4,'CA-900':4,'CA-945':5,'CO-802':3,'FL-331':2,'IL-606':2,'MA-021':2,'MI-481':2,'NC-275':1,'NJ-070':2,'TX-750':2,'TX-770':2,'UT-841':4,'WA-980':5}
BINS=[0,50,150,300,600,1000,1400,1800,np.inf]; ZLAB=['1','2','3','4','5','6','7','8']

# Task 1 is the overall U.S. market (Tsukumo + competitors), not Tsukumo-only demand.
base['MarketAnnualUnits']=MARKET_ANNUAL*base.PMF
base['TsukumoAnnualUnits']=TS_ANNUAL*base.PMF

# Y-A trend: current market (Year 0) through Year 5.
# Assumptions: 7.5% annual compound market growth and constant USD 3,000 unit price.
BASE_YEAR=2026
usa_summary=pd.DataFrame({
    'Combination':['Y-A-units / Y-A-amount']*6,
    'PlanningYear':['Year '+str(i) for i in range(6)],
    'CalendarYear':[BASE_YEAR+i for i in range(6)],
    'Geography':['USA']*6,
    'AnnualUnits':[CURRENT_MARKET*((1+MARKET_GROWTH)**i) for i in range(6)]
})
usa_summary['AnnualDollars']=usa_summary['AnnualUnits']*PRICE
usa_summary['AnnualWeightLb']=usa_summary['AnnualUnits']*WEIGHT
usa_summary['AnnualVolumeFt3']=usa_summary['AnnualUnits']*VOLUME
market_summary=base.groupby('Market',as_index=False).agg(PMF=('PMF','sum'),AnnualUnits=('MarketAnnualUnits','sum'))
market_summary['Combination']='Y-B-#'
market_summary['AnnualDollars']=market_summary.AnnualUnits*PRICE
market_summary['AnnualWeightLb']=market_summary.AnnualUnits*WEIGHT
market_summary['AnnualVolumeFt3']=market_summary.AnnualUnits*VOLUME
state_summary=base.groupby('State',as_index=False).agg(PMF=('PMF','sum'),AnnualUnits=('MarketAnnualUnits','sum')).sort_values('AnnualUnits',ascending=False)
state_summary['Combination']='Y-C-#'
zip_summary=base[['ZIP3','Market','State','Lat','Lon','PMF','MarketAnnualUnits']].copy()
zip_summary['Combination']='Y-D-#'
zip_summary=zip_summary.sort_values('MarketAnnualUnits',ascending=False)

def draw_us_boundaries(ax):
    """Draw a dark U.S. exterior impression and light state boundaries for the contiguous U.S."""
    for code,st in US_STATES.items():
        if code in ('AK','HI','PR'): continue
        ax.plot(st['lons'],st['lats'],color='0.72',linewidth=0.65,zorder=1)
    # A second pass emphasizes coastal/international state edges without obscuring ZIP3 data.
    coastal={'WA','OR','CA','AZ','NM','TX','LA','MS','AL','FL','GA','SC','NC','VA','MD','DE','NJ','NY','CT','RI','MA','NH','ME','VT','MI','MN','ND','MT','ID'}
    for code in coastal:
        st=US_STATES.get(code)
        if st: ax.plot(st['lons'],st['lats'],color='0.18',linewidth=1.25,zorder=2)
    ax.set_xlim(-125,-66); ax.set_ylim(24,50)
    ax.set_aspect('equal',adjustable='box')

def scatter_map(df,c,title,path,cmap='tab20',categorical=True,s=14):
    fig,ax=plt.subplots(figsize=(12,7))
    draw_us_boundaries(ax)
    if categorical:
        cats=pd.Categorical(df[c]); vals=cats.codes
        sc=ax.scatter(df.Lon,df.Lat,c=vals,cmap=cmap,s=s,alpha=.85,zorder=3)
        handles=[]
        for i,name in enumerate(cats.categories[:20]):
            handles.append(plt.Line2D([0],[0],marker='o',linestyle='',label=str(name),markerfacecolor=sc.cmap(sc.norm(i)),markeredgecolor='none',markersize=6))
        if len(cats.categories)<=20: ax.legend(handles=handles,bbox_to_anchor=(1.02,1),loc='upper left',fontsize=8)
    else:
        sc=ax.scatter(df.Lon,df.Lat,c=df[c],cmap=cmap,s=s,alpha=.85,zorder=3); fig.colorbar(sc,ax=ax,label=c)
    ax.set(title=title,xlabel='Longitude',ylabel='Latitude'); ax.grid(alpha=.12); fig.tight_layout(); fig.savefig(path,dpi=220); plt.close(fig)

# Task 1 submission-ready visualizations for the four selected combinations.
fig,ax1=plt.subplots(figsize=(10,6))
x=usa_summary['CalendarYear']
units=usa_summary['AnnualUnits']
amount=usa_summary['AnnualDollars']
line1=ax1.plot(x,units,marker='o',linewidth=2.2,label='Units')
ax1.set_xlabel('Year')
ax1.set_ylabel('Annual market demand (units)')
ax1.ticklabel_format(style='plain',axis='y')
ax1.grid(alpha=.20)
for yr,val in zip(x,units):
    ax1.annotate(f'{val/1_000_000:.2f}M',(yr,val),textcoords='offset points',xytext=(0,8),ha='center',fontsize=8)
ax2=ax1.twinx()
line2=ax2.plot(x,amount,marker='s',linestyle='--',linewidth=2.0,label='Amount (USD)')
ax2.set_ylabel('Annual market amount (USD)')
ax2.ticklabel_format(style='plain',axis='y')
for yr,val in zip(x,amount):
    amount_label=f'USD {val/1_000_000_000:.2f}B'
    ax2.annotate(amount_label,(yr,val),textcoords='offset points',xytext=(0,-15),ha='center',fontsize=8)
lines=line1+line2
ax1.legend(lines,[ln.get_label() for ln in lines],loc='upper left')
ax1.set_title('Task 1 - USA Units and Amount | Current to Year 5')
fig.tight_layout()
fig.savefig(FIG/'task1_Y-A-units_amount_USA_5year_trend.png',dpi=220)
plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5))
ms=market_summary.sort_values('AnnualUnits',ascending=False)
ax.bar(ms.Market,ms.AnnualUnits)
ax.set(title='Task 1 - Y-B-# | Annual Demand by Market Type',xlabel='Market type',ylabel='Annual demand (units)')
for i,v in enumerate(ms.AnnualUnits): ax.text(i,v,f'{v:,.0f}',ha='center',va='bottom',fontsize=9)
fig.tight_layout(); fig.savefig(FIG/'task1_Y-B-units_market_type.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(10,11))
ss=state_summary.sort_values('AnnualUnits',ascending=True)
ax.barh(ss.State,ss.AnnualUnits)
ax.set(title='Task 1 - Y-C-# | Annual Demand by State',xlabel='Annual demand (units)',ylabel='State')
fig.tight_layout(); fig.savefig(FIG/'task1_Y-C-units_state.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(13,8))
draw_us_boundaries(ax)
sc=ax.scatter(base.Lon,base.Lat,c=base.MarketAnnualUnits,cmap='viridis',
              s=10+180*(base.PMF/base.PMF.max()),alpha=.80,zorder=3)
fig.colorbar(sc,ax=ax,label='Annual market demand (units)')
ax.set(title='Task 1 - Y-D-# | Annual Demand by ZIP3',xlabel='Longitude',ylabel='Latitude')
ax.grid(alpha=.12); fig.tight_layout(); fig.savefig(FIG/'task1_Y-D-units_ZIP3_map.png',dpi=220); plt.close(fig)

# Market-type reference map retained for interpretation.
scatter_map(base,'Market','Task 1 - Market Type by ZIP3 Centroid',FIG/'task1_market_map.png','viridis',True)

def inv_tri(u,a,m,b):
    fc=(m-a)/(b-a); return np.where(u<fc,a+np.sqrt(u*(b-a)*(m-a)), b-np.sqrt((1-u)*(b-a)*(b-m)))
N=25
mg=inv_tri(rng.random(N),.04,.075,.12); sg=inv_tri(rng.random(N),.15,.20,.25)
annual_scen=CURRENT_MARKET*(1+mg)*SHARE*(1+sg)
week=pd.to_numeric(season['Proportion'].str.rstrip('%'))/100
day=pd.to_numeric(season['Proportion.1'].dropna().str.rstrip('%'))/100
scenario_daily_fc=[]; scenario_daily_net=[]; scenario_zip_daily=[]
base['ClosestFC']=base[FC15].idxmin(axis=1); base['ClosestMiles']=base[FC15].min(axis=1); base['ClosestZone']=pd.cut(base.ClosestMiles,BINS,labels=ZLAB,include_lowest=True,right=True).astype(str)
for k in range(N):
    wf=np.clip(week.values*(1+rng.normal(0,.20,len(week))),1e-9,None); wf/=wf.sum()
    df=np.clip(day.values*(1+rng.normal(0,.15,len(day))),1e-9,None); df/=df.sum()
    zp=np.clip(base.PMF.values*(1+rng.normal(0,.15,len(base))),0,None); zp/=zp.sum()
    dnet=[]; dfc={fc:[] for fc in FC15}; dzip=[]
    for w in range(52):
        for dow in range(7):
            total=annual_scen[k]*wf[w]*df[dow]
            dz=total*zp; dnet.append(dz.sum()); dzip.append(dz.copy())
            for fc in FC15: dfc[fc].append(dz[base.ClosestFC.values==fc].sum())
    scenario_daily_net.append(dnet); scenario_daily_fc.append(dfc); scenario_zip_daily.append(np.vstack(dzip))
scenario_daily_net=np.array(scenario_daily_net)
mean_daily=scenario_daily_net.mean(0); sd_daily=scenario_daily_net.std(0,ddof=1)
scen_stats={'min':float(annual_scen.min()),'mode_proxy':float(np.median(annual_scen)),'mean':float(annual_scen.mean()),'std':float(annual_scen.std(ddof=1)),'max':float(annual_scen.max())}
for name,z in [('68',1),('95',1.65),('99',2.33)]: scen_stats[name+'_upper']=scen_stats['mean']+z*scen_stats['std']

# Task 2 submission-ready tables and figures.
task2_scenarios=pd.DataFrame({
    'Scenario':np.arange(1,N+1),
    'MarketGrowth':mg,
    'TsukumoShareGrowth':sg,
    'AnnualTsukumoDemand':annual_scen
})
task2_daily=pd.DataFrame({'Day':np.arange(1,DAYS+1),'MeanDailyDemand':mean_daily,'StdDemand':sd_daily})
for lab,z in [('68',1.0),('95',1.65),('99',2.33)]:
    task2_daily[f'RobustUpper{lab}']=mean_daily+z*sd_daily
task2_scenarios.to_csv(OUT/'task2_scenarios.csv',index=False)
task2_daily.to_csv(OUT/'task2_daily_robust_demand.csv',index=False)

fig,ax=plt.subplots(figsize=(9,5.5))
ax.hist(annual_scen,bins=8,alpha=.65,edgecolor='black')
ax.axvline(scen_stats['mean'],linestyle='-',linewidth=2,label=f"Mean = {scen_stats['mean']:,.0f}")
for lab,ls in [('68','--'),('95','-.'),('99',':')]:
    ax.axvline(scen_stats[lab+'_upper'],linestyle=ls,linewidth=1.8,label=f"{lab}% upper = {scen_stats[lab+'_upper']:,.0f}")
ax.set(title='Task 2 - Annual Tsukumo Demand Scenario Distribution',xlabel='Annual demand (units)',ylabel='Scenario count')
ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(FIG/'task2_annual_scenario_distribution.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(12,5.5))
ax.plot(task2_daily.Day,task2_daily.MeanDailyDemand,label='Mean daily demand',linewidth=1.4)
ax.plot(task2_daily.Day,task2_daily.RobustUpper68,label='68% robust upper',linewidth=1.0,alpha=.8)
ax.plot(task2_daily.Day,task2_daily.RobustUpper95,label='95% robust upper',linewidth=1.0,alpha=.8)
ax.plot(task2_daily.Day,task2_daily.RobustUpper99,label='99% robust upper',linewidth=1.2)
ax.set(title='Task 2 - Daily Seasonal Robust Demand',xlabel='Planning day',ylabel='Tsukumo demand (units/day)')
ax.legend(ncol=2); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(FIG/'task2_daily_seasonal_robust_demand.png',dpi=220); plt.close(fig)

# Task 3 submission-ready 1-FC / 4-FC / 15-FC closest-FC and shipment-zone outputs.
TASK3_CONFIGS={
    '1-FC':['GA-303'],
    '4-FC':['GA-303','NY-134','TX-799','UT-841'],
    '15-FC':FC15
}
task3_summary_rows=[]; task3_zone_frames=[]
for cfg_name,fcs in TASK3_CONFIGS.items():
    cfg=base.copy()
    cfg['ClosestFC']=cfg[fcs].idxmin(axis=1)
    cfg['ClosestMiles']=cfg[fcs].min(axis=1)
    cfg['ClosestZone']=pd.cut(cfg.ClosestMiles,BINS,labels=ZLAB,include_lowest=True,right=True).astype(str)
    cfg['AnnualTsukumoUnits']=cfg.PMF*TS_ANNUAL
    cfg['WeightedMiles']=cfg.PMF*cfg.ClosestMiles
    prefix=cfg_name.replace('-','').lower()
    fc_sum=cfg.groupby('ClosestFC',as_index=False).agg(DemandShare=('PMF','sum'))
    fc_sum['AnnualUnits']=fc_sum.DemandShare*TS_ANNUAL
    market_zone=pd.pivot_table(cfg,index='Market',columns='ClosestZone',values='PMF',aggfunc='sum',fill_value=0)
    market_zone=market_zone.div(market_zone.sum(axis=1),axis=0).reset_index()
    zone_share=cfg.groupby('ClosestZone',as_index=False,observed=False).agg(DemandShare=('PMF','sum'))
    zone_share['Configuration']=cfg_name
    cfg[['ZIP3','Market','State','Lat','Lon','PMF','ClosestFC','ClosestMiles','ClosestZone','AnnualTsukumoUnits']].to_csv(OUT/f'task3_{prefix}_zip_assignment.csv',index=False)
    fc_sum.to_csv(OUT/f'task3_{prefix}_fc_summary.csv',index=False)
    market_zone.to_csv(OUT/f'task3_{prefix}_distance_market.csv',index=False)
    zone_share.to_csv(OUT/f'task3_{prefix}_zone_share.csv',index=False)
    scatter_map(cfg,'ClosestFC',f'Task 3 - {cfg_name} Closest-FC ZIP3 Clusters',FIG/f'task3_{cfg_name}_closest_fc_map.png','tab20',True)
    task3_summary_rows.append([cfg_name,len(fcs),float(cfg.WeightedMiles.sum()),float(cfg.loc[cfg.ClosestZone.astype(int)<=3,'PMF'].sum()),float(cfg.loc[cfg.ClosestZone.astype(int)>=6,'PMF'].sum())])
    task3_zone_frames.append(zone_share)
task3_config_summary=pd.DataFrame(task3_summary_rows,columns=['Configuration','FCCount','DemandWeightedMiles','DemandShareZone1to3','DemandShareZone6to8'])
task3_zone_comparison=pd.concat(task3_zone_frames,ignore_index=True)
task3_config_summary.to_csv(OUT/'task3_configuration_summary.csv',index=False)
task3_zone_comparison.to_csv(OUT/'task3_zone_share_comparison.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True)
for ax,(cfg_name,grp) in zip(axes,task3_zone_comparison.groupby('Configuration',sort=False)):
    g=grp.copy(); g['ClosestZone']=g['ClosestZone'].astype(int); g=g.sort_values('ClosestZone')
    ax.bar(g['ClosestZone'].astype(str),g.DemandShare*100)
    ax.set_title(cfg_name); ax.set_xlabel('Shipment Zone'); ax.grid(axis='y',alpha=.15)
axes[0].set_ylabel('National demand share (%)')
fig.suptitle('Task 3 - PMF-Weighted Shipment-Zone Distribution by FC Configuration')
fig.tight_layout(); fig.savefig(FIG/'task3_zone_distribution_comparison.png',dpi=220); plt.close(fig)

fc_summary=base.groupby('ClosestFC',as_index=False).agg(DemandShare=('PMF','sum')); fc_summary['AnnualUnits']=fc_summary.DemandShare*TS_ANNUAL
fc_market=pd.pivot_table(base,index='ClosestFC',columns='Market',values='PMF',aggfunc='sum',fill_value=0).reset_index()
dist_market=pd.pivot_table(base,index='Market',columns='ClosestZone',values='PMF',aggfunc='sum',fill_value=0)
dist_market=dist_market.div(dist_market.sum(axis=1),axis=0).reset_index()

def build_task4(cfg_name,fcs):
    cfg=base.copy()
    cfg['ClosestFC']=cfg[fcs].idxmin(axis=1)
    cfg['ClosestMiles']=cfg[fcs].min(axis=1)
    cfg['ClosestZone']=pd.cut(cfg.ClosestMiles,BINS,labels=ZLAB,include_lowest=True,right=True).astype(str)
    def eligible(row):
        maxz=min(8,int(row['ClosestZone'])+1)
        return tuple(sorted(fc for fc in fcs if int(pd.cut(pd.Series([row[fc]]),BINS,labels=ZLAB,include_lowest=True,right=True).iloc[0])<=maxz))
    cfg['FulfillmentSet']=cfg.apply(eligible,axis=1)
    cfg['FCCount']=cfg.FulfillmentSet.str.len()
    cfg['Cluster']=cfg.FulfillmentSet.apply(lambda x:'|'.join(x))
    cfg['SourceType']=np.where(cfg.FCCount==1,'Single-source','Multi-source')
    alloc=[]
    for _,r in cfg.iterrows():
        cands=list(r.FulfillmentSet); close=r.ClosestFC
        if len(cands)==1:
            alloc.append((r.ZIP3,close,1.0,r.PMF,r.Market,r[close]))
        else:
            alloc.append((r.ZIP3,close,.8,r.PMF*.8,r.Market,r[close]))
            others=[x for x in cands if x!=close]
            for fc in others:
                alloc.append((r.ZIP3,fc,.2/len(others),r.PMF*.2/len(others),r.Market,r[fc]))
    alloc=pd.DataFrame(alloc,columns=['ZIP3','FC','Allocation','AllocatedPMF','Market','Miles'])
    alloc['Zone']=pd.cut(alloc.Miles,BINS,labels=ZLAB,include_lowest=True,right=True).astype(str)
    alloc['AnnualTsukumoUnits']=alloc.AllocatedPMF*TS_ANNUAL
    single=float(cfg.loc[cfg.FCCount==1,'PMF'].sum()); multi=1-single
    source_summary=pd.DataFrame([{'Configuration':cfg_name,'SingleSourceShare':single,'MultiSourceShare':multi}])
    alloc_dist=pd.pivot_table(alloc,index='Market',columns='Zone',values='AllocatedPMF',aggfunc='sum',fill_value=0)
    alloc_dist=alloc_dist.div(alloc_dist.sum(axis=1),axis=0).reset_index()
    prefix=cfg_name.replace('-','').lower()
    cfg[['ZIP3','Market','State','Lat','Lon','PMF','ClosestFC','ClosestMiles','ClosestZone','FulfillmentSet','FCCount','Cluster','SourceType']].to_csv(OUT/f'task4_{prefix}_zip_clusters.csv',index=False)
    alloc.to_csv(OUT/f'task4_{prefix}_allocation.csv',index=False)
    source_summary.to_csv(OUT/f'task4_{prefix}_source_summary.csv',index=False)
    alloc_dist.to_csv(OUT/f'task4_{prefix}_allocation_zone_distribution.csv',index=False)
    # Cluster maps can have many categories; for Task 4 place the complete legend
    # outside the plotting area on the right so no cluster labels are omitted.
    fig,ax=plt.subplots(figsize=(15,7))
    draw_us_boundaries(ax)
    cats=pd.Categorical(cfg['Cluster']); vals=cats.codes
    sc=ax.scatter(cfg.Lon,cfg.Lat,c=vals,cmap='tab20',s=12,alpha=.85,zorder=3)
    handles=[plt.Line2D([0],[0],marker='o',linestyle='',label=str(name),
             markerfacecolor=sc.cmap(sc.norm(i)),markeredgecolor='none',markersize=6)
             for i,name in enumerate(cats.categories)]
    ax.legend(handles=handles,title='Fulfillment Cluster',loc='center left',
              bbox_to_anchor=(1.01,.5),fontsize=7,title_fontsize=8,
              frameon=True,ncol=max(1,math.ceil(len(handles)/28)))
    ax.set(title=f'Task 4 - {cfg_name} Fulfillment Clusters',xlabel='Longitude',ylabel='Latitude')
    fig.subplots_adjust(right=.73)
    fig.savefig(FIG/f'task4_{cfg_name}_fulfillment_clusters.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    scatter_map(cfg,'FCCount',f'Task 4 - {cfg_name} Number of Eligible FCs by ZIP3',FIG/f'task4_{cfg_name}_fc_count_map.png','RdYlGn',False,s=16)
    return cfg,alloc,source_summary,alloc_dist

task4_results={}
task4_source_frames=[]; task4_alloc_frames=[]
for cfg_name,fcs in TASK3_CONFIGS.items():
    cfg4,alloc4,source4,dist4=build_task4(cfg_name,fcs)
    task4_results[cfg_name]=(cfg4,alloc4,source4,dist4)
    task4_source_frames.append(source4)
    d=dist4.copy(); d.insert(0,'Configuration',cfg_name); task4_alloc_frames.append(d)
task4_source_comparison=pd.concat(task4_source_frames,ignore_index=True)
task4_allocation_zone_comparison=pd.concat(task4_alloc_frames,ignore_index=True)
task4_source_comparison.to_csv(OUT/'task4_source_comparison.csv',index=False)
task4_allocation_zone_comparison.to_csv(OUT/'task4_allocation_zone_comparison.csv',index=False)

# Submission figure: eligible-FC count maps for all three network configurations.
fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharex=True,sharey=True)
for ax,(cfg_name,(cfg4,_,_,_)) in zip(axes,task4_results.items()):
    draw_us_boundaries(ax)
    sc=ax.scatter(cfg4.Lon,cfg4.Lat,c=cfg4.FCCount,cmap='viridis',s=12,alpha=.85,zorder=3)
    ax.set_title(cfg_name); ax.set_xlabel('Longitude')
axes[0].set_ylabel('Latitude')
fig.colorbar(sc,ax=axes.ravel().tolist(),label='Eligible FC count',shrink=.82)
fig.suptitle('Task 4 - Number of Eligible FCs by ZIP3')
fig.subplots_adjust(left=.05,right=.92,bottom=.10,top=.86,wspace=.08)
fig.savefig(FIG/'task4_fc_count_comparison.png',dpi=220); plt.close(fig)

# Submission figure: single-source vs multi-source PMF-weighted demand share.
fig,ax=plt.subplots(figsize=(8,5))
x=np.arange(len(task4_source_comparison)); w=.36
ax.bar(x-w/2,task4_source_comparison.SingleSourceShare*100,w,label='Single-source')
ax.bar(x+w/2,task4_source_comparison.MultiSourceShare*100,w,label='Multi-source')
ax.set_xticks(x,task4_source_comparison.Configuration); ax.set_ylabel('National demand share (%)')
ax.set_title('Task 4 - Single-Source vs Multi-Source Demand Share'); ax.legend(); ax.grid(axis='y',alpha=.15)
fig.tight_layout(); fig.savefig(FIG/'task4_source_share_comparison.png',dpi=220); plt.close(fig)

# Keep 15-FC objects as downstream defaults for Tasks 5-10.
base,alloc,_,alloc_dist=task4_results['15-FC']
single_share=float(base.loc[base.FCCount==1,'PMF'].sum())

conv={'Primary':[1,.9,.75,.6,.4,.3],'Secondary':[1,1,.95,.75,.6,.4],'Tertiary':[1,1,1,.95,.8,.6]}; promises=['1','2','3','4','5','5+']
ship=np.array([[607,353,230,139,121,103],[759,441,287,173,151,128],[1025,585,392,238,191,160],[1445,794,533,316,242,198],[2078,924,655,343,287,225],[2692,1427,895,491,340,259],[2841,1795,1202,776,362,276],[2912,1854,1330,894,388,301]])

# Task 5: delivery-promise economics using the Task 3 closest-FC assignment,
# as explicitly required by the casework. Task 4's 80/20 multi-source allocation
# is not used for Task 5 customer-delivery economics.
task5_optimal_rows=[]; task5_config_rows=[]
for cfg_name,(cfg5,alloc5,_,_) in task4_results.items():
    econ5=[]
    for mt in ['Primary','Secondary','Tertiary']:
        sub=cfg5[cfg5.Market==mt].copy()
        potential_units=TS_ANNUAL*sub.PMF.sum()
        for j,promise in enumerate(promises):
            c=conv[mt][j]
            units=potential_units*c; revenue=units*PRICE
            sc=(TS_ANNUAL*sub.PMF*c*sub.ClosestZone.astype(int).map(lambda z:ship[z-1,j])).sum()
            cogs=units*COGS; net=revenue-sc; gp=net-cogs
            econ5.append([mt,promise,potential_units,c,units,revenue,sc,cogs,net,gp])
    econ5=pd.DataFrame(econ5,columns=['Market','PromiseDays','PotentialUnits','ConversionRate','ConvertedUnits','Revenue','ShippingCost','COGS','NetRevenue','GrossProfit'])
    opt5=econ5.loc[econ5.groupby('Market').GrossProfit.idxmax()].copy()
    optimized5=opt5[['ConvertedUnits','Revenue','ShippingCost','COGS','NetRevenue','GrossProfit']].sum()
    prefix=cfg_name.replace('-','').lower()
    config5=pd.DataFrame([{'Configuration':cfg_name,
        'PrimaryPromiseDays':str(opt5.loc[opt5.Market=='Primary','PromiseDays'].iloc[0]),
        'SecondaryPromiseDays':str(opt5.loc[opt5.Market=='Secondary','PromiseDays'].iloc[0]),
        'TertiaryPromiseDays':str(opt5.loc[opt5.Market=='Tertiary','PromiseDays'].iloc[0]),
        'ConvertedUnits':float(optimized5.ConvertedUnits),'Revenue':float(optimized5.Revenue),
        'ShippingCost':float(optimized5.ShippingCost),'COGS':float(optimized5.COGS),
        'NetRevenue':float(optimized5.NetRevenue),'GrossProfit':float(optimized5.GrossProfit)}])
    econ5.to_csv(OUT/f'task5_{prefix}_delivery_promise_economics.csv',index=False)
    opt5.to_csv(OUT/f'task5_{prefix}_optimal_delivery_promise.csv',index=False)
    config5.to_csv(OUT/f'task5_{prefix}_configuration_economics.csv',index=False)
    task5_optimal_rows.append(opt5.assign(Configuration=cfg_name)); task5_config_rows.append(config5)
    fig,ax=plt.subplots(figsize=(9,5))
    piv=econ5.pivot(index='PromiseDays',columns='Market',values='GrossProfit').reindex(promises)
    piv.plot(kind='bar',ax=ax); ax.set(xlabel='Delivery promise (days)',ylabel='Gross profit ($)',title=f'Task 5 - {cfg_name} Gross Profit by Delivery Promise')
    ax.grid(axis='y',alpha=.15); fig.tight_layout(); fig.savefig(FIG/f'task5_{prefix}_gross_profit_by_promise.png',dpi=220); plt.close(fig)

task5_optimal_all=pd.concat(task5_optimal_rows,ignore_index=True)
task5_optimal_summary=task5_optimal_all[['Configuration','Market','PromiseDays','ConvertedUnits','Revenue','ShippingCost','COGS','GrossProfit']]
task5_optimal_summary.to_csv(OUT/'task5_optimal_delivery_promise_summary.csv',index=False)
task5_configuration_economics=pd.concat(task5_config_rows,ignore_index=True)
task5_configuration_economics.to_csv(OUT/'task5_configuration_economics_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5)); promise_num={'1':1,'2':2,'3':3,'4':4,'5':5,'5+':6}
for mt in ['Primary','Secondary','Tertiary']:
    sub=task5_optimal_all[task5_optimal_all.Market==mt]
    ax.plot(sub.Configuration,[promise_num[str(x)] for x in sub.PromiseDays],marker='o',label=mt)
ax.set_yticks([1,2,3,4,5,6],['1','2','3','4','5','5+'])
ax.set(xlabel='FC configuration',ylabel='Optimal delivery promise (days)',title='Task 5 - Optimal Delivery Promise by Network Configuration')
ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(FIG/'task5_optimal_promise_comparison.png',dpi=220); plt.close(fig)

# Task 6: optimize OTD promise separately for each market type for all three FC configurations.
# Exhaustive enumeration is exact here because the objective is additive across market types
# and Task 6 introduces no cross-market capacity constraint.
task6_network_rows=[]
for cfg_name,(cfg6,_,_,_) in task4_results.items():
    prefix=cfg_name.replace('-','').lower()
    econ6=[]
    for mt in ['Primary','Secondary','Tertiary']:
        sub=cfg6[cfg6.Market==mt].copy()
        potential_units=TS_ANNUAL*sub.PMF.sum()
        for j,promise in enumerate(promises):
            c=conv[mt][j]
            units=potential_units*c; revenue=units*PRICE
            sc=(TS_ANNUAL*sub.PMF*c*sub.ClosestZone.astype(int).map(lambda z:ship[z-1,j])).sum()
            cogs=units*COGS; net=revenue-sc; gp=net-cogs
            econ6.append([mt,promise,potential_units,c,units,revenue,sc,cogs,net,gp])
    econ6=pd.DataFrame(econ6,columns=['Market','PromiseDays','PotentialUnits','ConversionRate','ConvertedUnits','Revenue','ShippingCost','COGS','NetRevenue','GrossProfit'])
    opt6=econ6.loc[econ6.groupby('Market').GrossProfit.idxmax()].copy()
    sum_cols=['PotentialUnits','ConvertedUnits','Revenue','ShippingCost','COGS','NetRevenue','GrossProfit']
    totals6=opt6[sum_cols].sum()
    overall6=pd.DataFrame([{'Market':'Overall','PromiseDays':'Market-specific',**{c:float(totals6[c]) for c in sum_cols}}])
    pd.concat([opt6,overall6],ignore_index=True,sort=False).to_csv(OUT/f'{prefix}_task6_optimal.csv',index=False)
    policy_detail=[]; policy_rows=[]
    for label,promise in [('Optimized',None),('1-day all','1'),('5+ day all','5+')]:
        x=opt6.copy() if promise is None else econ6[econ6.PromiseDays==promise].copy()
        x.insert(0,'Policy',label); policy_detail.append(x)
        t=x[sum_cols].sum(); policy_rows.append([label,*[float(t[c]) for c in sum_cols]])
    pd.concat(policy_detail,ignore_index=True).to_csv(OUT/f'{prefix}_task6_policy_market_detail.csv',index=False)
    policy6=pd.DataFrame(policy_rows,columns=['Policy',*sum_cols])
    policy6.to_csv(OUT/f'{prefix}_task6_policy_compare.csv',index=False)
    task6_network_rows.append([cfg_name,
        str(opt6.loc[opt6.Market=='Primary','PromiseDays'].iloc[0]),
        str(opt6.loc[opt6.Market=='Secondary','PromiseDays'].iloc[0]),
        str(opt6.loc[opt6.Market=='Tertiary','PromiseDays'].iloc[0]),
        float(totals6.GrossProfit)])
    fig,ax=plt.subplots(figsize=(9,5)); piv=econ6.pivot(index='PromiseDays',columns='Market',values='GrossProfit').reindex(promises)
    piv.plot(kind='bar',ax=ax); ax.set(xlabel='OTD promise (days)',ylabel='Gross operating profit ($)',title=f'Task 6 - {cfg_name} Gross Profit by OTD Promise')
    ax.grid(axis='y',alpha=.15); fig.tight_layout(); fig.savefig(FIG/f'task6_{prefix}_gross_profit_by_otd.png',dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(policy6.Policy,policy6.GrossProfit)
    ax.set(xlabel='Policy',ylabel='Gross operating profit ($)',title=f'Task 6 - {cfg_name} Policy Comparison')
    ax.grid(axis='y',alpha=.15); fig.tight_layout(); fig.savefig(FIG/f'task6_{prefix}_policy_comparison.png',dpi=220); plt.close(fig)

task6_network=pd.DataFrame(task6_network_rows,columns=['Configuration','PrimaryOTD','SecondaryOTD','TertiaryOTD','OptimizedGrossProfit'])
task6_network.to_csv(OUT/'task6_network_optimized_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5)); ax.bar(task6_network.Configuration,task6_network.OptimizedGrossProfit)
ax.set(title='Task 6 - Optimized Gross Profit by FC Configuration',ylabel='Gross operating profit ($)')
ax.grid(axis='y',alpha=.15); fig.tight_layout(); fig.savefig(FIG/'task6_network_optimized_gross_profit.png',dpi=220); plt.close(fig)

# Keep the 15-FC Task 5 objects as downstream defaults for existing Tasks 6-10.
econ=task5_optimal_all[task5_optimal_all.Configuration=='15-FC'].copy()
opt=econ.copy()
optimized=opt[['ConvertedUnits','Revenue','ShippingCost','NetRevenue','GrossProfit']].sum()

# Task 7: robust autonomy targeting for 1-FC, 4-FC, and 15-FC.
# FC-level stochastic demand uses the Task 4 fulfillment allocation (80% to the
# closest FC and 20% split across other eligible FCs for multi-source ZIP3s).
# Each FC maintains a 3-week / 99% robust autonomy target. The DC carries the
# incremental stock required to reach the overall network autonomy target.
def forward_robust(mu,sd,L,z):
    n=len(mu); out=np.zeros(n)
    for t in range(n):
        ix=np.arange(t,t+L)%n
        out[t]=mu[ix].sum()+z*np.sqrt((sd[ix]**2).sum())
    return out

task7_config_rows=[]; task7_results={}
zip_index={z:i for i,z in enumerate(base.ZIP3.astype(str))}
for cfg_name,(cfg7,alloc7,_,_) in task4_results.items():
    fcs7=TASK3_CONFIGS[cfg_name]; prefix=cfg_name.replace('-','').lower()
    W=np.zeros((len(base),len(fcs7)))
    fc_index={fc:j for j,fc in enumerate(fcs7)}
    for _,r in alloc7.iterrows():
        W[zip_index[str(r.ZIP3)],fc_index[r.FC]]+=float(r.Allocation)
    scen_fc=np.stack([arr @ W for arr in scenario_zip_daily],axis=0)  # scenario x day x FC
    fc_mean7=scen_fc.mean(axis=0); fc_sd7=scen_fc.std(axis=0,ddof=1)
    fc_target7=np.column_stack([forward_robust(fc_mean7[:,j],fc_sd7[:,j],21,2.33) for j in range(len(fcs7))])
    fc_total7=fc_target7.sum(axis=1)

    fc_daily=pd.DataFrame({'Day':np.arange(1,DAYS+1)})
    for j,fc in enumerate(fcs7):
        fc_daily[f'{fc}_MeanDailyDemand']=fc_mean7[:,j]
        fc_daily[f'{fc}_StdDailyDemand']=fc_sd7[:,j]
        fc_daily[f'{fc}_TargetInventory_3wk99']=fc_target7[:,j]
    fc_daily.to_csv(OUT/f'task7_{prefix}_fc_daily_inventory.csv',index=False)

    fc_max=pd.DataFrame({
        'FC':fcs7,
        'MaxTargetInventory_3wk99':[float(fc_target7[:,j].max()) for j in range(len(fcs7))],
        'AverageTargetInventory_3wk99':[float(fc_target7[:,j].mean()) for j in range(len(fcs7))]
    })
    fc_max.to_csv(OUT/f'task7_{prefix}_fc_max_inventory.csv',index=False)

    robust_rows=[]; scenario_profiles={}
    for Lw in [4,6,8]:
        for lab,z in [('50',0),('68',1),('95',1.65),('99',2.33)]:
            nt=forward_robust(mean_daily,sd_daily,Lw*7,z)
            dc=np.maximum(nt-fc_total7,0)
            scenario_profiles[(Lw,lab)]=(nt,dc)
            robust_rows.append([Lw,lab,float(fc_total7.max()),float(dc.max()),float(nt.max())])
    robust7=pd.DataFrame(robust_rows,columns=['Weeks','Robustness','MaxFCInventory','MaxDCInventory','MaxNetworkInventory'])
    robust7.to_csv(OUT/f'task7_{prefix}_robustness_comparison.csv',index=False)

    net_target7,dc_target7=scenario_profiles[(6,'99')]
    network_daily=pd.DataFrame({
        'Day':np.arange(1,DAYS+1),
        'TotalFCInventory_3wk99':fc_total7,
        'DCInventory_6wk99':dc_target7,
        'TotalNetworkInventory_6wk99':net_target7
    })
    network_daily.to_csv(OUT/f'task7_{prefix}_network_daily_inventory.csv',index=False)

    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot(network_daily.Day,network_daily.TotalFCInventory_3wk99,label='Total FC inventory (3-week, 99%)')
    ax.plot(network_daily.Day,network_daily.DCInventory_6wk99,label='DC inventory (increment to 6-week, 99%)')
    ax.plot(network_daily.Day,network_daily.TotalNetworkInventory_6wk99,label='Network inventory (6-week, 99%)')
    ax.set(xlabel='Day',ylabel='Units',title=f'Task 7 - {cfg_name} Daily Robust Inventory Targets')
    ax.legend(); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(FIG/f'task7_{prefix}_daily_inventory_profile.png',dpi=220); plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5))
    for lab in ['50','68','95','99']:
        q=robust7[robust7.Robustness==lab]
        ax.plot(q.Weeks,q.MaxNetworkInventory,marker='o',label=f'{lab}%')
    ax.set(xticks=[4,6,8],xlabel='Network autonomy period (weeks)',ylabel='Maximum network inventory (units)',
           title=f'Task 7 - {cfg_name} Autonomy / Robustness Comparison')
    ax.legend(title='Robustness'); ax.grid(alpha=.15); fig.tight_layout(); fig.savefig(FIG/f'task7_{prefix}_autonomy_robustness.png',dpi=220); plt.close(fig)

    task7_config_rows.append([cfg_name,float(fc_total7.max()),float(dc_target7.max()),float(net_target7.max())])
    task7_results[cfg_name]={'fc_daily':fc_daily,'fc_max':fc_max,'robustness':robust7,'network_daily':network_daily}

task7_configuration_summary=pd.DataFrame(task7_config_rows,columns=['Configuration','MaxFCInventory_3wk99','MaxDCInventory_6wk99','MaxNetworkInventory_6wk99'])
task7_configuration_summary.to_csv(OUT/'task7_configuration_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(8,5)); x=np.arange(len(task7_configuration_summary)); w=.25
ax.bar(x-w,task7_configuration_summary.MaxFCInventory_3wk99,w,label='FC total')
ax.bar(x,task7_configuration_summary.MaxDCInventory_6wk99,w,label='DC')
ax.bar(x+w,task7_configuration_summary.MaxNetworkInventory_6wk99,w,label='Network')
ax.set_xticks(x,task7_configuration_summary.Configuration); ax.set_ylabel('Maximum inventory (units)')
ax.set_title('Task 7 - Base 3-week FC / 6-week Network, 99% Robustness'); ax.legend(); ax.grid(axis='y',alpha=.15)
fig.tight_layout(); fig.savefig(FIG/'task7_configuration_inventory_comparison.png',dpi=220); plt.close(fig)

# Tasks 8-10: run downstream planning for 1-FC, 4-FC, and 15-FC.
# Task 8 AP production requirement is network-wide and therefore common in demand terms,
# while DC inventory differs slightly by configuration because aggregate FC robust stock differs.
q0,a0,r0,e0=.98,.90,.95,.90; handling_min=12.0; shift_min=480.0
units_per_resource=(shift_min/handling_min)*q0*a0*r0*e0
STORAGE_OM=6.60; STORAGE_SETUP=46.20; SEASONAL_PCT=95.0

def round_f(x):
    lo=math.floor(x); return lo if x-lo<=.3 else math.ceil(x)

def backlog_duration(x):
    return int(np.count_nonzero(np.asarray(x)>1e-9))

def storage_tier_cost(profile):
    x=np.asarray(profile,dtype=float); base_units=float(x.min()); p95=float(np.percentile(x,SEASONAL_PCT)); peak_units=float(x.max())
    seasonal_units=max(0.0,p95-base_units); peak_extreme_units=max(0.0,peak_units-p95)
    base_use=np.minimum(x,base_units); seasonal_use=np.minimum(np.maximum(x-base_units,0.0),seasonal_units); peak_use=np.maximum(x-base_units-seasonal_units,0.0)
    def pos_increase_cost(u):
        inc=np.maximum(np.diff(np.r_[0.0,u]),0.0); return float(inc.sum()),float(inc.sum()*STORAGE_SETUP)
    b_inc,b_setup=pos_increase_cost(base_use); s_inc,s_setup=pos_increase_cost(seasonal_use); p_inc,p_setup=pos_increase_cost(peak_use)
    b_om=float(base_use.sum()*STORAGE_OM); s_om=float(seasonal_use.sum()*STORAGE_OM); p_om=float(peak_use.sum()*STORAGE_OM)
    return [base_units,seasonal_units,peak_extreme_units,peak_units,float(base_use.sum()),float(seasonal_use.sum()),float(peak_use.sum()),b_om,s_om,p_om,b_inc,s_inc,p_inc,b_setup,s_setup,p_setup,b_om+s_om+p_om,b_setup+s_setup+p_setup,b_om+s_om+p_om+b_setup+s_setup+p_setup]

task8_all=[]; task9_all=[]; task10_all=[]; downstream_results={}
for cfg_name in ['1-FC','4-FC','15-FC']:
    prefix=cfg_name.replace('-','').lower(); fcs=TASK3_CONFIGS[cfg_name]
    fc_daily7=task7_results[cfg_name]['fc_daily']; net_daily7=task7_results[cfg_name]['network_daily']
    fc_mean_cfg={fc:fc_daily7[f'{fc}_MeanDailyDemand'].to_numpy() for fc in fcs}
    fc_sd_cfg={fc:fc_daily7[f'{fc}_StdDailyDemand'].to_numpy() for fc in fcs}
    fc_target_cfg={fc:fc_daily7[f'{fc}_TargetInventory_3wk99'].to_numpy() for fc in fcs}
    net_target_cfg=net_daily7.TotalNetworkInventory_6wk99.to_numpy(); dc_target_cfg=net_daily7.DCInventory_6wk99.to_numpy()

    # Task 8
    target_change_cfg=np.diff(net_target_cfg,prepend=net_target_cfg[-1]); pursuit_cfg=np.maximum(0,mean_daily+target_change_cfg)
    def simulate_production_cfg(rate,initial_extra=0.0):
        bal=float(initial_extra); inv=np.zeros(DAYS); back=np.zeros(DAYS)
        for t in range(DAYS):
            bal += float(rate[t])-float(pursuit_cfg[t])
            if bal>=0: inv[t]=bal
            else: back[t]=-bal
        return inv,back
    const_cfg=float(pursuit_cfg.sum()/DAYS); full_rate_cfg=np.repeat(const_cfg,DAYS)
    full_inv_cfg,full_back_cfg=simulate_production_cfg(full_rate_cfg,0.0); full_pre_cfg=float(full_back_cfg.max()); full_pre_inv_cfg,_=simulate_production_cfg(full_rate_cfg,full_pre_cfg)
    segments=[(0,91),(91,182),(182,273),(273,DAYS)]; seg_cfg=np.zeros(DAYS)
    for a,b in segments: seg_cfg[a:b]=pursuit_cfg[a:b].mean()
    _,seg_back0_cfg=simulate_production_cfg(seg_cfg,0.0); seg_pre_cfg=float(seg_back0_cfg.max()); seg_inv_cfg,_=simulate_production_cfg(seg_cfg,seg_pre_cfg)
    prod_compare_cfg=pd.DataFrame([
      ['Pursuit',float(pursuit_cfg.max()),0.0,0,0.0,float(dc_target_cfg.max())],
      ['Full smoothing - no preproduction',const_cfg,float(full_back_cfg.max()),backlog_duration(full_back_cfg),0.0,float((dc_target_cfg+full_inv_cfg).max())],
      ['Full smoothing - with preproduction',const_cfg,0.0,0,full_pre_cfg,float((dc_target_cfg+full_pre_inv_cfg).max())],
      ['Segmented smoothing (4 x 13 weeks)',float(seg_cfg.max()),0.0,0,seg_pre_cfg,float((dc_target_cfg+seg_inv_cfg).max())]
    ],columns=['Strategy','MaxDailyProduction','MaxBacklog','BacklogDays','Preproduction','MaxDCInventory'])
    prod_compare_cfg.insert(0,'Configuration',cfg_name); prod_compare_cfg.to_csv(OUT/f'task8_{prefix}_production_strategy_comparison.csv',index=False); task8_all.append(prod_compare_cfg)
    pd.DataFrame({'Day':np.arange(1,DAYS+1),'MeanDemand':mean_daily,'NetworkTarget_6wk99':net_target_cfg,'PursuitRequirement':pursuit_cfg,'FullSmoothingRate':full_rate_cfg,'FullSmoothingWithPreproductionInventory':full_pre_inv_cfg,'SegmentedSmoothingRate':seg_cfg,'SegmentedSmoothingInventory':seg_inv_cfg}).to_csv(OUT/f'task8_{prefix}_daily_production_profiles.csv',index=False)

    # Task 9
    d99_cfg={fc:fc_mean_cfg[fc]+2.33*fc_sd_cfg[fc] for fc in fcs}
    def simulate_fc_cfg(fc,interval=7,threshold=14):
        demand=d99_cfg[fc]; target=fc_target_cfg[fc]; lead=RAD[fc]; onhand=float(target[0]); pipeline=[]; last_ship=-interval; rows=[]
        for t in range(DAYS):
            arrivals=sum(q for day,q in pipeline if day==t); onhand+=arrivals; pipeline=[x for x in pipeline if x[0]>t]; onhand=max(0.0,onhand-float(demand[t])); position=onhand+sum(q for _,q in pipeline)
            cum=0.0; autonomy=0
            for h in range(1,22):
                cum+=float(demand[(t+h)%DAYS])
                if position+1e-9>=cum: autonomy=h
                else: break
            due=(t-last_ship)>=interval; breach=autonomy<threshold; qty=0.0; floored=False
            if (breach or due) and autonomy<21:
                required=max(float(target[t])-position,0.0); qty=max(required,27.0); floored=required<27.0; arrival_day=t+lead
                if lead==0: onhand+=qty; position+=qty; arrivals+=qty
                elif arrival_day<DAYS: pipeline.append((arrival_day,qty))
                last_ship=t
            rows.append([t+1,onhand,position,autonomy,qty,floored,arrivals,float(demand[t])])
        return pd.DataFrame(rows,columns=['Day','OnHand','InventoryPosition','RobustAutonomyDays','ReplenishmentQty','Floored27','Arrivals','OutboundDemand'])
    sims_cfg={fc:simulate_fc_cfg(fc) for fc in fcs}; replen_rows=[]; fc_rows=[]
    for fc,sfc in sims_cfg.items():
        sfc.to_csv(OUT/f'task9_{prefix}_{fc}_daily_replenishment.csv',index=False); orders=sfc[sfc.ReplenishmentQty>0]
        replen_rows.append([fc,len(orders),float(orders.ReplenishmentQty.mean()) if len(orders) else 0,float(orders.ReplenishmentQty.max()) if len(orders) else 0,float(orders.ReplenishmentQty.sum()),int(sfc.Floored27.sum())])
        daily_move=sfc.Arrivals+sfc.OutboundDemand; fc_rows.append([fc,'FC',float(daily_move.max()),int(sfc.loc[daily_move.idxmax(),'Day'])])
    pd.DataFrame(replen_rows,columns=['FC','ShipmentCount','AverageShipmentUnits','MaxShipmentUnits','AnnualReplenishmentUnits','Floored27Count']).to_csv(OUT/f'task9_{prefix}_replenishment_summary.csv',index=False)
    dc_outbound_cfg=sum((sfc.ReplenishmentQty.to_numpy() for sfc in sims_cfg.values()),np.zeros(DAYS))
    ap_profiles_cfg={'Pursuit':pursuit_cfg,'Full smoothing with pre-production':full_rate_cfg,'Segmented smoothing':seg_cfg}; dc_rows=[]
    for strategy,dc_inbound in ap_profiles_cfg.items():
        dc_daily=dc_inbound+dc_outbound_cfg; dc_rows.append([strategy,float(dc_inbound.max()),float(dc_outbound_cfg.max()),float(dc_daily.max()),int(np.argmax(dc_daily)+1)])
    dc_strategy_cfg=pd.DataFrame(dc_rows,columns=['ProductionStrategy','MaxDCInboundUnits','MaxDCOutboundUnits','MaxDailyDCThroughputUnits','PeakDay']); dc_strategy_cfg.to_csv(OUT/f'task9_{prefix}_dc_throughput_by_production_strategy.csv',index=False)
    resource_rows=[]
    for fac,basis,peak,day in fc_rows:
        raw=peak/units_per_resource; n=round_f(raw); resource_rows.append([fac,basis,peak,day,raw,n,max(0.0,peak-n*units_per_resource)])
    for _,rr in dc_strategy_cfg.iterrows():
        peak=float(rr.MaxDailyDCThroughputUnits); raw=peak/units_per_resource; n=round_f(raw); resource_rows.append(['DC-GA-303',rr.ProductionStrategy,peak,int(rr.PeakDay),raw,n,max(0.0,peak-n*units_per_resource)])
    thr_cfg=pd.DataFrame(resource_rows,columns=['Facility','Basis','MaxDailyThroughputUnits','PeakDay','RawResources','Resources','ResidualUnitsIfRoundedDown']); thr_cfg.insert(0,'Configuration',cfg_name); thr_cfg.to_csv(OUT/f'task9_{prefix}_throughput_resources.csv',index=False); task9_all.append(thr_cfg)
    storage_rows=[]
    for fc,sfc in sims_cfg.items(): storage_rows.append([fc,*storage_tier_cost(sfc.OnHand.to_numpy())])
    retained_dc_cfg=dc_target_cfg+full_pre_inv_cfg; storage_rows.append(['DC-GA-303',*storage_tier_cost(retained_dc_cfg)])
    storage_cols=['Facility','BaseCapacityUnits','SeasonalCapacityUnits','PeakExtremeCapacityUnits','PeakInventoryUnits','BaseUnitDays','SeasonalUnitDays','PeakExtremeUnitDays','BaseOMCost','SeasonalOMCost','PeakExtremeOMCost','BaseSetupIncreaseUnits','SeasonalSetupIncreaseUnits','PeakExtremeSetupIncreaseUnits','BaseSetupCost','SeasonalSetupCost','PeakExtremeSetupCost','TotalOMCost','TotalSetupCost','TotalStorageCost']
    pd.DataFrame(storage_rows,columns=storage_cols).to_csv(OUT/f'task9_{prefix}_storage_tier_costs.csv',index=False)

    # Task 10
    alloc_cfg=task4_results[cfg_name][1].merge(base[['ZIP3','PMF']],on='ZIP3',how='left'); demand_share_cfg=alloc_cfg.assign(Weighted=lambda d:d.Allocation*d.PMF).groupby('FC').Weighted.sum().to_dict()
    t10=[]
    for fc in fcs:
        threshold=min(20,RAD[fc]+3); max_interval=max(1,21-threshold); candidates=[]
        for interval in range(1,max_interval+1):
            cand=simulate_fc_cfg(fc,interval,threshold); orders=cand[cand.ReplenishmentQty>0]; floor_rate=float(orders.Floored27.mean()) if len(orders) else 0.0
            candidates.append((interval,floor_rate,float(cand.OnHand.mean()),len(orders),cand))
        feasible=[x for x in candidates if x[1]<=0.05]
        interval,floor_rate,avg_inv,shipments,proposed=(min(feasible,key=lambda x:(x[2],x[1],x[0])) if feasible else min(candidates,key=lambda x:(x[1],x[2],x[0])))
        uniform=sims_cfg[fc]; uorders=uniform[uniform.ReplenishmentQty>0]; porders=proposed[proposed.ReplenishmentQty>0]; median_qty=float(porders.ReplenishmentQty.median()) if len(porders) else 0.0
        max_typical_qty=float(d99_cfg[fc].mean()*max_interval); viable=bool(max_typical_qty>=27.0)
        t10.append([fc,float(demand_share_cfg.get(fc,0)),RAD[fc],float(d99_cfg[fc].mean()),threshold,interval,max_interval,'Interval',shipments,int(proposed.Floored27.sum()),floor_rate,median_qty,avg_inv,float(uniform.OnHand.mean()),int(uniform.Floored27.sum()),float(uorders.Floored27.mean()) if len(uorders) else 0.0,viable,max_typical_qty])
        proposed.to_csv(OUT/f'task10_{prefix}_{fc}_proposed_daily.csv',index=False)
    t10_cfg=pd.DataFrame(t10,columns=['FC','DemandShare','RADdays','AvgRobustDailyDemand','ThresholdDays','IntervalDays','ThresholdTriggerDay','GoverningTrigger','ShipmentCount','Floored27Count','FloorRate','MedianShipmentQty','ProposedAvgInventory','UniformAvgInventory','UniformFloored27Count','UniformFloorRate','FTLMinimumViable','MaxTypicalQtyBeforeThreshold'])
    t10_cfg['AvgInventoryReductionPct']=100*(t10_cfg.UniformAvgInventory-t10_cfg.ProposedAvgInventory)/t10_cfg.UniformAvgInventory; t10_cfg.insert(0,'Configuration',cfg_name); t10_cfg.to_csv(OUT/f'task10_{prefix}_fc_policy_summary.csv',index=False); task10_all.append(t10_cfg)

prod_compare=pd.concat(task8_all,ignore_index=True); prod_compare.to_csv(OUT/'task8_all_configurations_production_strategy_comparison.csv',index=False)
thr=pd.concat(task9_all,ignore_index=True); thr.to_csv(OUT/'task9_all_configurations_throughput_resources.csv',index=False)
task10=pd.concat(task10_all,ignore_index=True); task10.to_csv(OUT/'task10_all_configurations_fc_policy_summary.csv',index=False)
task10comp=task10.copy(); task10comp.to_csv(OUT/'task10_all_configurations_policy_comparison.csv',index=False)

# Submission-ready figures for Tasks 8-10.
# Keep all generated PNG files in the same tsukumo_outputs/figures directory used by Tasks 1-7.
# Task 8: production strategy capacity comparison across network configurations.
fig,ax=plt.subplots(figsize=(9,5))
strategies=list(prod_compare.Strategy.drop_duplicates())
x=np.arange(len(strategies)); width=.24
for i,cfg_name in enumerate(['1-FC','4-FC','15-FC']):
    d=prod_compare[prod_compare.Configuration==cfg_name].set_index('Strategy').reindex(strategies)
    ax.bar(x+(i-1)*width,d.MaxDailyProduction,width,label=cfg_name)
ax.set_xticks(x,strategies,rotation=15,ha='right'); ax.set_ylabel('Maximum daily AP production (units/day)')
ax.set_title('Task 8 - AP Production Capacity by Strategy and Configuration'); ax.legend(); ax.grid(axis='y',alpha=.15)
fig.tight_layout(); fig.savefig(FIG/'task8_production_strategy_capacity_comparison.png',dpi=220,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5))
for cfg_name in ['1-FC','4-FC','15-FC']:
    d=prod_compare[prod_compare.Configuration==cfg_name]
    ax.plot(d.Strategy,d.MaxDCInventory,marker='o',label=cfg_name)
ax.set_ylabel('Maximum DC inventory (units)'); ax.set_title('Task 8 - Maximum DC Inventory by Production Strategy')
ax.tick_params(axis='x',rotation=15); ax.legend(); ax.grid(alpha=.15)
fig.tight_layout(); fig.savefig(FIG/'task8_dc_inventory_by_strategy.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Task 9: FC throughput and DC throughput under the alternative AP strategies.
fig,ax=plt.subplots(figsize=(10,6))
fc_thr=thr[thr.Basis=='FC'].copy()
order=fc_thr.groupby('Facility').MaxDailyThroughputUnits.max().sort_values(ascending=True).index
for cfg_name in ['1-FC','4-FC','15-FC']:
    d=fc_thr[fc_thr.Configuration==cfg_name].set_index('Facility').reindex(order)
    ax.scatter(d.MaxDailyThroughputUnits,order,label=cfg_name,s=42)
ax.set_xlabel('Peak daily FC throughput (units/day)'); ax.set_title('Task 9 - Peak FC Throughput by Configuration')
ax.legend(); ax.grid(alpha=.15)
fig.tight_layout(); fig.savefig(FIG/'task9_fc_peak_throughput_all_configurations.png',dpi=220,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(9,5)); dc_thr=thr[thr.Facility=='DC-GA-303'].copy()
strategies_dc=list(dc_thr.Basis.drop_duplicates()); x=np.arange(len(strategies_dc)); width=.24
for i,cfg_name in enumerate(['1-FC','4-FC','15-FC']):
    d=dc_thr[dc_thr.Configuration==cfg_name].set_index('Basis').reindex(strategies_dc)
    ax.bar(x+(i-1)*width,d.MaxDailyThroughputUnits,width,label=cfg_name)
ax.set_xticks(x,strategies_dc,rotation=15,ha='right'); ax.set_ylabel('Peak DC throughput (units/day)')
ax.set_title('Task 9 - DC Peak Throughput by AP Production Strategy'); ax.legend(); ax.grid(axis='y',alpha=.15)
fig.tight_layout(); fig.savefig(FIG/'task9_dc_peak_throughput_by_strategy.png',dpi=220,bbox_inches='tight'); plt.close(fig)

# Task 10: FC-specific interval/threshold policy and inventory effect.
fig,ax=plt.subplots(figsize=(11,6)); t10_15=task10[task10.Configuration=='15-FC'].sort_values('DemandShare',ascending=False)
x=np.arange(len(t10_15)); width=.38
ax.bar(x-width/2,t10_15.IntervalDays,width,label='Replenishment interval')
ax.bar(x+width/2,t10_15.ThresholdDays,width,label='Minimum robust autonomy')
ax.set_xticks(x,t10_15.FC,rotation=45,ha='right'); ax.set_ylabel('Days')
ax.set_title('Task 10 - 15-FC Replenishment Interval and Robust Autonomy Threshold'); ax.legend(); ax.grid(axis='y',alpha=.15)
fig.tight_layout(); fig.savefig(FIG/'task10_15fc_interval_threshold_policy.png',dpi=220,bbox_inches='tight'); plt.close(fig)

fig,ax=plt.subplots(figsize=(11,6)); x=np.arange(len(t10_15)); width=.38
ax.bar(x-width/2,t10_15.UniformAvgInventory,width,label='Uniform 7-day / 14-day policy')
ax.bar(x+width/2,t10_15.ProposedAvgInventory,width,label='FC-specific proposed policy')
ax.set_xticks(x,t10_15.FC,rotation=45,ha='right'); ax.set_ylabel('Average on-hand inventory (units)')
ax.set_title('Task 10 - Average FC Inventory: Uniform vs Proposed Policy'); ax.legend(); ax.grid(axis='y',alpha=.15)
fig.tight_layout(); fig.savefig(FIG/'task10_15fc_inventory_policy_comparison.png',dpi=220,bbox_inches='tight'); plt.close(fig)
robust=task7_results['15-FC']['robustness'].copy(); econ=task5_optimal_all[task5_optimal_all.Configuration=='15-FC'].copy(); sensitivity=pd.DataFrame()
outputs={'task1_usa_summary':usa_summary,'task1_market_summary':market_summary,'task1_state_summary':state_summary,'task1_zip_summary':zip_summary,'task2_scenario_stats':pd.DataFrame([scen_stats]),'task3_fc_summary':fc_summary,'task3_fc_market':fc_market,'task3_distance_market':dist_market,'task4_zip_clusters':base[['ZIP3','Lat','Lon','ClosestFC','ClosestZone','Cluster','FCCount','PMF']],'task4_alloc_distance':alloc_dist,'task5_economics':econ,'task7_robustness':robust,'task8_production':prod_compare,'task9_throughput':thr,'task9_sensitivity':sensitivity,'task10_policy':task10,'task10_compare':task10comp}
for name,df in outputs.items(): df.to_csv(OUT/f'{name}.csv',index=False)
summary={'market_annual':MARKET_ANNUAL,'tsukumo_annual':TS_ANNUAL,'single_fc_demand_share':single_share,'scenario_stats':scen_stats,'optimized_policy':opt[['Market','PromiseDays']].to_dict('records'),'optimized_totals':optimized.to_dict(),'units_per_resource':units_per_resource,'missing_storage_rates_note':'Storage tiers use the instructed common $6.60/unit/day O&M rate. The $46.20/unit setup charge applies only to positive capacity increases, with no credit for decreases; the P95 split is a reporting convention because identical tier rates make the cost-minimizing label split non-unique.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
