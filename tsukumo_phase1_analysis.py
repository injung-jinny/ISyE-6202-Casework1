from pathlib import Path
import math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.sampledata.us_states import data as US_STATES

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
RAD={'GA-303':0,'AZ-852':4,'CA-900':4,'CA-945':5,'CO-802':3,'FL-331':2,'IL-606':2,'MA-021':2,'MI-481':2,'NC-275':1,'NJ-070':2,'TX-750':2,'TX-770':2,'UT-841':4,'WA-980':5}
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
scenario_daily_fc=[]; scenario_daily_net=[]
base['ClosestFC']=base[FC15].idxmin(axis=1); base['ClosestMiles']=base[FC15].min(axis=1); base['ClosestZone']=pd.cut(base.ClosestMiles,BINS,labels=ZLAB,include_lowest=True,right=True).astype(str)
for k in range(N):
    wf=np.clip(week.values*(1+rng.normal(0,.20,len(week))),1e-9,None); wf/=wf.sum()
    df=np.clip(day.values*(1+rng.normal(0,.15,len(day))),1e-9,None); df/=df.sum()
    zp=np.clip(base.PMF.values*(1+rng.normal(0,.15,len(base))),0,None); zp/=zp.sum()
    dnet=[]; dfc={fc:[] for fc in FC15}
    for w in range(52):
        for dow in range(7):
            total=annual_scen[k]*wf[w]*df[dow]
            dz=total*zp; dnet.append(dz.sum())
            for fc in FC15: dfc[fc].append(dz[base.ClosestFC.values==fc].sum())
    scenario_daily_net.append(dnet); scenario_daily_fc.append(dfc)
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
task2_daily=pd.DataFrame({'Day':np.arange(1,DAYS+1),'MeanDemand':mean_daily,'StdDemand':sd_daily})
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
ax.plot(task2_daily.Day,task2_daily.MeanDemand,label='Mean daily demand',linewidth=1.4)
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
econ=[]
for mt in ['Primary','Secondary','Tertiary']:
    sub=base[base.Market==mt]
    for j,p in enumerate(promises):
        c=conv[mt][j]; units=TS_ANNUAL*sub.PMF.sum()*c; revenue=units*PRICE
        sc=(TS_ANNUAL*sub.PMF*c*sub.ClosestZone.astype(int).map(lambda z:ship[z-1,j])).sum()
        net=revenue-sc; gp=net-units*COGS
        econ.append([mt,p,units,revenue,sc,net,gp])
econ=pd.DataFrame(econ,columns=['Market','PromiseDays','ConvertedUnits','Revenue','ShippingCost','NetRevenue','GrossProfit'])
opt=econ.loc[econ.groupby('Market').GrossProfit.idxmax()].copy(); optimized=opt[['ConvertedUnits','Revenue','ShippingCost','NetRevenue','GrossProfit']].sum()
def policy_total(p): return econ[econ.PromiseDays==p][['ConvertedUnits','Revenue','ShippingCost','NetRevenue','GrossProfit']].sum()
policy_compare=pd.DataFrame([['Optimized',*optimized.values],['1-day all',*policy_total('1').values],['5+ all',*policy_total('5+').values]],columns=['Policy','ConvertedUnits','Revenue','ShippingCost','NetRevenue','GrossProfit'])
fig,ax=plt.subplots(figsize=(9,5)); piv=econ.pivot(index='PromiseDays',columns='Market',values='GrossProfit').reindex(promises); piv.plot(kind='bar',ax=ax); ax.set_ylabel('Gross profit ($)'); ax.set_title('Task 5 - Gross profit by OTD promise and market type'); fig.tight_layout(); fig.savefig(FIG/'task5_gross_profit.png',dpi=180); plt.close(fig)

fc_mean={fc:np.mean(np.array([x[fc] for x in scenario_daily_fc]),axis=0) for fc in FC15}; fc_sd={fc:np.std(np.array([x[fc] for x in scenario_daily_fc]),axis=0,ddof=1) for fc in FC15}
def forward_robust(mu,sd,L,z):
    n=len(mu); out=np.zeros(n)
    for t in range(n):
        ix=np.arange(t,t+L)%n; out[t]=mu[ix].sum()+z*np.sqrt((sd[ix]**2).sum())
    return out
fc_target={fc:forward_robust(fc_mean[fc],fc_sd[fc],21,2.33) for fc in FC15}
fc_total=np.sum(np.vstack([fc_target[f] for f in FC15]),axis=0)
robust_rows=[]; network_targets={}
for Lw in [4,6,8]:
  for lab,z in [('50',0),('68',1),('95',1.65),('99',2.33)]:
    nt=forward_robust(mean_daily,sd_daily,Lw*7,z); network_targets[(Lw,lab)]=nt
    dc=np.maximum(nt-fc_total,0)
    robust_rows.append([Lw,lab,float(fc_total.max()),float(dc.max()),float(nt.max())])
robust=pd.DataFrame(robust_rows,columns=['Weeks','Robustness','MaxFCInventory','MaxDCInventory','MaxNetworkInventory'])

net_target=network_targets[(6,'99')]; dc_target=np.maximum(net_target-fc_total,0)
pursuit=np.maximum(0,mean_daily+np.r_[net_target[0],np.diff(net_target)])
const=float(pursuit.sum()/DAYS)
cum_gap=np.cumsum(pursuit-const); backlog=np.maximum(cum_gap,0); preprod=float(max(0,cum_gap.max()))
seg=np.zeros(DAYS)
for a,b in [(0,91),(91,182),(182,273),(273,364)]: seg[a:b]=pursuit[a:b].mean()
seg_gap=np.cumsum(pursuit-seg); seg_pre=float(max(0,seg_gap.max()))
prod_compare=pd.DataFrame([
 ['Pursuit',pursuit.max(),0,0,dc_target.max()],
 ['Full smoothing - no preproduction',const,backlog.max(),0,dc_target.max()],
 ['Full smoothing - with preproduction',const,0,preprod,dc_target.max()+preprod],
 ['Segmented smoothing (4)',seg.max(),0,seg_pre,dc_target.max()+seg_pre]],columns=['Strategy','MaxDailyProduction','MaxBacklog','Preproduction','MaxDCInventory'])
fig,ax=plt.subplots(figsize=(11,5)); ax.plot(pursuit,label='Pursuit',alpha=.75); ax.plot(np.repeat(const,DAYS),label='Full smoothing'); ax.plot(seg,label='Segmented'); ax.legend(); ax.set(title='Task 8 - Production strategy profiles',xlabel='Day',ylabel='Units/day'); fig.tight_layout(); fig.savefig(FIG/'task8_production.png',dpi=180); plt.close(fig)

d99={fc:fc_mean[fc]+2.33*fc_sd[fc] for fc in FC15}
def simulate_fc(fc,interval=7,threshold=14):
    demand=d99[fc]; target=fc_target[fc]; lead=RAD[fc]; onhand=target[0]; pipeline=[]; last_ship=-interval; rows=[]
    for t in range(DAYS):
        arrivals=sum(q for day,q in pipeline if day==t); onhand+=arrivals; pipeline=[x for x in pipeline if x[0]>t]
        onhand=max(0,onhand-demand[t]); position=onhand+sum(q for _,q in pipeline)
        cum=0; autonomy=0
        for h in range(1,22):
            cum+=demand[(t+h)%DAYS]
            if position+1e-9>=cum: autonomy=h
            else: break
        due=(t-last_ship)>=interval; breach=autonomy<threshold
        qty=0; floored=False
        if (breach or due) and autonomy<21:
            qty=max(target[t]-position,27); floored=qty<=27+1e-9; pipeline.append((min(t+lead,DAYS-1),qty)); last_ship=t
        rows.append([t,onhand,position,autonomy,qty,floored,arrivals,demand[t]])
    return pd.DataFrame(rows,columns=['Day','OnHand','InventoryPosition','RobustAutonomyDays','ReplenishmentQty','Floored27','Arrivals','OutboundDemand'])
sims={fc:simulate_fc(fc) for fc in FC15}
thr=[]
for fc in FC15:
    s=sims[fc]; req=float((s.Arrivals+s.OutboundDemand).max()); thr.append([fc,req])
thr=pd.DataFrame(thr,columns=['Facility','MaxDailyThroughputUnits'])
outdc=np.zeros(DAYS)
for fc,s in sims.items(): outdc+=s.ReplenishmentQty.values
dc_req=float((outdc+const).max()); thr=pd.concat([thr,pd.DataFrame([['DC-GA-303',dc_req]],columns=thr.columns)],ignore_index=True)
eff_per_resource=40*.98*.90*.95*.90
thr['RawResources']=thr.MaxDailyThroughputUnits/eff_per_resource
def round_f(x):
    lo=math.floor(x); return lo if x-lo<=.3 else math.ceil(x)
thr['Resources']=thr.RawResources.map(round_f)
thr['ResidualUnitsIfRoundedDown']=np.maximum(0,thr.MaxDailyThroughputUnits-thr.Resources*eff_per_resource)
sens=[]
for var,vals in [('q',[.93,.98,1]),('r',[.90,.95,.99]),('e',[.85,.90,.95])]:
 for val in vals:
  q=.98;r=.95;e=.90
  if var=='q':q=val
  if var=='r':r=val
  if var=='e':e=val
  cap=40*q*.90*r*e; n=sum(round_f(x/cap) for x in thr.MaxDailyThroughputUnits); annual=n*3.6*364+n*25.2
  sens.append([var,val,cap,n,annual])
sensitivity=pd.DataFrame(sens,columns=['Parameter','Value','UnitsPerResourceDay','TotalResources','AnnualOMPlusSetup'])
params=[]; comp=[]
for fc in FC15:
    avg=fc_mean[fc].mean(); interval=int(min(14,max(7,math.ceil(35/max(avg,1e-9)))))
    threshold=min(20,RAD[fc]+3)
    s=simulate_fc(fc,interval,threshold); u=sims[fc]
    params.append([fc,float(base.loc[base.ClosestFC==fc,'PMF'].sum()),RAD[fc],interval,threshold,float(s.OnHand.mean()),int(s.Floored27.sum()),int((s.ReplenishmentQty>0).sum())])
    comp.append([fc,float(u.OnHand.mean()),int(u.Floored27.sum()),float(s.OnHand.mean()),int(s.Floored27.sum())])
task10=pd.DataFrame(params,columns=['FC','DemandShare','RADdays','IntervalDays','ThresholdDays','AvgInventory','Floored27Count','ShipmentCount'])
task10comp=pd.DataFrame(comp,columns=['FC','UniformAvgInventory','UniformFloored27','ProposedAvgInventory','ProposedFloored27'])
outputs={'task1_usa_summary':usa_summary,'task1_market_summary':market_summary,'task1_state_summary':state_summary,'task1_zip_summary':zip_summary,'task2_scenario_stats':pd.DataFrame([scen_stats]),'task3_fc_summary':fc_summary,'task3_fc_market':fc_market,'task3_distance_market':dist_market,'task4_zip_clusters':base[['ZIP3','Lat','Lon','ClosestFC','ClosestZone','Cluster','FCCount','PMF']],'task4_alloc_distance':alloc_dist,'task5_economics':econ,'task6_optimal':opt,'task6_policy_compare':policy_compare,'task7_robustness':robust,'task8_production':prod_compare,'task9_throughput':thr,'task9_sensitivity':sensitivity,'task10_policy':task10,'task10_compare':task10comp}
for name,df in outputs.items(): df.to_csv(OUT/f'{name}.csv',index=False)
summary={'market_annual':MARKET_ANNUAL,'tsukumo_annual':TS_ANNUAL,'single_fc_demand_share':single_share,'scenario_stats':scen_stats,'optimized_policy':opt[['Market','PromiseDays']].to_dict('records'),'optimized_totals':optimized.to_dict(),'one_day':policy_total('1').to_dict(),'five_plus':policy_total('5+').to_dict(),'eff_per_resource':eff_per_resource,'missing_storage_rates_note':'Appendix 1 supplies only base storage O&M and setup rates; seasonal and peak-and-extreme rates are not numerically supplied, so lowest-cost tier optimization cannot be completed without class-provided rates.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
