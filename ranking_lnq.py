#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: reubendo
"""

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
from scipy.stats import kendalltau, rankdata

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker



def rank_based_on_score(serie, list_teams):
    df_output = {'Team':list_teams, 'Rank':[]}
    
    for team in list_teams:
        v_team = serie.T[team]
        rank = 1 + sum([v<v_team for v in serie.values])
        df_output['Rank'].append(rank)
        
    return pd.DataFrame(df_output).set_index('Team')

def iqr_numpy(d, v=1):
    Q1 = np.quantile(d, 0.25)
    Q3 = np.quantile(d, 0.75)
    return [Q1,Q3]

METRICS = ['Dice', 'ASSD']
DICE_METRICS = [m for m in METRICS if 'Dice' in  m]
ASSD_METRICS = [m for m in METRICS if 'ASSD' in  m]

METRICS_NAME = {
    'Dice':'Lymph Node - Dice (%)', 
    'ASSD':'Lymph Node - ASSD (mm)', 
}

np.random.seed(8102023) #Day of the conference


# 0/ Load the scores
df_scores = pd.read_csv('phase3_results.csv')

df_scores['Team']= df_scores['Team'].str.replace('Wlty','Hilab')
df_scores['Team']= df_scores['Team'].str.replace('stefanfischer','CompAI')
df_scores['Team']= df_scores['Team'].str.replace('1810929','Skeleton Suns')
df_scores['Team']= df_scores['Team'].str.replace('Larry.Zhang','IMR')


teams = sorted(set(df_scores['Team'].to_list()))
cases = sorted(set(df_scores['Case'].to_list()))

df_scores['ASSD'].fillna(df_scores['ASSD'].max(), inplace=True)

for team in teams:
    print(df_scores[df_scores['Team']==team].isnull().sum())

print(f"Number of cases: {len(cases)}")
print(f"Number of teams: {len(teams)}")

# 1/ Rank for each case, for each region, and for each measure
df_rank = {'Case':[], 'Team':[], 'Metric':[], 'Rank':[]}
for case in tqdm(cases):
    for metric in METRICS:
        # First find score
        all_values = dict()
        for team in teams:
            mask = (df_scores['Case']==case)&(df_scores['Team']==team)
            v_team_case = df_scores[mask][metric].values[0]
            all_values[team] = v_team_case
        if not np.all(np.isnan(np.array(list(all_values.values())))):
            if np.any(np.isnan(np.array(list(all_values.values())))):
                print('error code')
            # Second associate rank to team
            for team in teams:
                v_team_case = all_values[team]
                if 'Dice' in metric: # Using minimum rank
                    rank = 1 + sum([v>v_team_case for v in all_values.values()])
                else:
                    rank = 1 + sum([v<v_team_case for v in all_values.values()])
                
                df_rank['Case'].append(case)
                df_rank['Team'].append(team)
                df_rank['Metric'].append(metric)
                df_rank['Rank'].append(rank)
            
df_rank = pd.DataFrame(df_rank)

# 2/ Cumulative Rank for each case
df_cum_rank = {'Case':[], 
                      'Team':[], 
                      'CumRank_All':[],
                      'CumRank_Dice':[],
                      'CumRank_ASSD':[],}

for case in tqdm(cases):
    for team in teams:
        # Using all metrics
        mask_all = (df_rank['Case']==case)&(df_rank['Team']==team)
        cumulat_rank = df_rank[mask_all]['Rank'].mean()
        
        # Using Dice Score Coefficient only
        mask_dice = mask_all&(df_rank['Metric'].isin(DICE_METRICS))
        cumulat_rank_dice = df_rank[mask_dice]['Rank'].mean()
        
        # Using ASSD
        mask_assd = mask_all&(df_rank['Metric'].isin(ASSD_METRICS))
        cumulat_rank_assd = df_rank[mask_assd]['Rank'].mean()
        
        
        df_cum_rank['Case'].append(case)
        df_cum_rank['Team'].append(team)
        df_cum_rank['CumRank_All'].append(cumulat_rank)
        df_cum_rank['CumRank_Dice'].append(cumulat_rank_dice)
        df_cum_rank['CumRank_ASSD'].append(cumulat_rank_assd)

df_cum_rank = pd.DataFrame(df_cum_rank)

# 3/ Final Ranking
df_final = df_cum_rank.groupby(['Team']).mean()

ranked_teams = dict()
for cum_metric in ['CumRank_All', 'CumRank_Dice', 'CumRank_ASSD']:
    ranked_teams[cum_metric] = list(df_final.sort_values(cum_metric).index)
    
print('--- Ranking ---')
for i, t in enumerate(ranked_teams['CumRank_All']):
    print(f"{i+1}: {t}")
    
ranks_ranked_team = list(range(1,len(teams)+1))
color_ranked_team = sns.color_palette("hls", len(teams))


# Print scores for the top 3 teams
for metric in METRICS:
    df_metric = dict()
    #df_metric['cases'] = cases
    df_temp = df_scores[df_scores['Team'].isin(ranked_teams['CumRank_All'][:3])]
    # df_temp = df_temp[df_temp['Case'].isin(cases_center[center])]
    scores_team = df_temp[metric].tolist()
    scores_team = [min(100,k) for k in scores_team]
    if 'Dice' in metric:
        print(metric, np.round(np.median(scores_team)*100,1), [np.round(k*100,1) for k in iqr_numpy(scores_team)])
    else:
        print(metric, np.round(np.median(scores_team),1), np.round(iqr_numpy(scores_team), 2))


# 4/ Create and save Figures  
dict_df_metrics = dict()
for metric in METRICS:
    df_metric = dict()
    #df_metric['cases'] = cases
    for t in teams:
        df_temp = df_scores[df_scores['Team']==t]
        scores_team = df_temp[metric].tolist()
        scores_team = np.array(scores_team)
        scores_team = scores_team[~np.isnan(scores_team)]
        scores_team = [min(100,k) for k in scores_team]
        df_metric[t] = scores_team
        
    df_metric = pd.DataFrame(df_metric)
    dict_df_metrics[metric] = df_metric

# Create figure distribution results for each metric
for metric in METRICS:
    fig, ax = plt.subplots(1, 1,figsize=(7,7))
    sns.set_style("darkgrid")
    # ascending = True

    if 'Dice' in metric:
        ascending = False
        # pass
    else:
        ascending = True
        ax.set_xscale("log")
        
    team_order = dict_df_metrics[metric].median().sort_values(ascending=ascending).index.tolist()
    color_order = [color_ranked_team[ranked_teams['CumRank_All'].index(k)] for k in team_order]
    
    if 'Dice' in metric:
        factor = 100
    else:
        factor = 1
    df_melt = pd.melt(factor*dict_df_metrics[metric], var_name='Team', value_name=METRICS_NAME[metric])
    if 'Dice' in metric:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        # sns.boxplot(y="Team", x=METRICS_NAME[metric], palette=color_order, data=df_melt, orient='h', order=team_order).invert_xaxis()
        sns.boxplot(y="Team", x=METRICS_NAME[metric], palette=color_order, data=df_melt, orient='h', order=team_order)
    else:
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
        sns.boxplot(y="Team", x=METRICS_NAME[metric], palette=color_order, data=df_melt, orient='h', order=team_order)
    
    
    fig.savefig(f'imgs2023_lnq//{metric}.pdf', bbox_inches = 'tight', pad_inches = 0, dpi=100)


# Create figure for the cumulative ranks
fig, ax = plt.subplots(1, 1,figsize=(7,7))
sns.boxplot(y="Team",
            x='CumRank_All',
            palette=color_ranked_team, 
            data=df_cum_rank, 
            orient='h', 
            order=ranked_teams['CumRank_All']).set(
    xlabel='Cumulative Rank', 
)
fig.savefig('imgs2023_lnq/Cumulative_rank.pdf', bbox_inches = 'tight', pad_inches = 0, dpi=100)
                
fig, ax = plt.subplots(1, 1,figsize=(7,7))
sns.boxplot(y="Team",
            x='CumRank_Dice',
            palette=color_ranked_team, 
            data=df_cum_rank, 
            orient='h', 
            order=ranked_teams['CumRank_All']).set(
    xlabel='Cumulative Rank', 
)
                
fig, ax = plt.subplots(1, 1,figsize=(7,7))
sns.boxplot(y="Team",
            x='CumRank_ASSD',
            palette=color_ranked_team, 
            data=df_cum_rank, 
            orient='h', 
            order=ranked_teams['CumRank_All']).set(
    xlabel='Cumulative Rank', 
)               

                
# 6/ Comparision with other ranking techniques
df_ranking_method = []
# Rank-then-aggreagate with mean (ours)
df_rank_then_aggregate_mean_ranking = rank_based_on_score(df_rank.groupby(['Team']).mean()['Rank'], teams)
df_rank_then_aggregate_mean_ranking['Ranking Method'] = ['rank-then-mean (ours)']*len(teams)
df_ranking_method.append(df_rank_then_aggregate_mean_ranking)

# Rank-then-aggreagate with median (ours)
df_rank_then_aggregate_median_ranking = rank_based_on_score(df_rank.groupby(['Team']).median()['Rank'], teams)
df_rank_then_aggregate_median_ranking['Ranking Method'] = ['rank-then-median']*len(teams)
df_ranking_method.append(df_rank_then_aggregate_median_ranking)

# Aggregate-then-rank with mean
df_rank_then_aggregate_mean = {'Team':teams}
aggregate_scores_mean = df_scores.groupby(['Team']).mean()
for metric in METRICS:
    df_rank_then_aggregate_mean[metric] = []
    v_metric = aggregate_scores_mean[metric]
    for team in teams:
        v_team_case = v_metric.T[team]
        if 'Dice' in metric: # Using minimum rank
            rank = 1 + sum([v>v_team_case for v in v_metric.values])
        else:
            rank = 1 + sum([v<v_team_case for v in v_metric.values])
        df_rank_then_aggregate_mean[metric].append(rank)
        
df_rank_then_aggregate_mean = pd.DataFrame(df_rank_then_aggregate_mean).set_index('Team')
df_rank_then_aggregate_mean['mean'] = df_rank_then_aggregate_mean.mean(axis=1)

df_aggregate_then_rank_mean_ranking = rank_based_on_score(df_rank_then_aggregate_mean['mean'], teams)
df_aggregate_then_rank_mean_ranking['Ranking Method'] = ['mean-then-rank']*len(teams)
df_ranking_method.append(df_aggregate_then_rank_mean_ranking)


# Aggregate-then-rank with median
df_rank_then_aggregate_median = {'Team':teams}
aggregate_scores_median = df_scores.groupby(['Team']).median()
for metric in METRICS:
    df_rank_then_aggregate_median[metric] = []
    v_metric = aggregate_scores_median[metric]
    for team in teams:
        v_team_case = v_metric.T[team]
        if 'Dice' in metric: # Using minimum rank
            rank = 1 + sum([v>v_team_case for v in v_metric.values])
        else:
            rank = 1 + sum([v<v_team_case for v in v_metric.values])
        df_rank_then_aggregate_median[metric].append(rank)
        
df_rank_then_aggregate_median = pd.DataFrame(df_rank_then_aggregate_median).set_index('Team')
df_rank_then_aggregate_median['median'] = df_rank_then_aggregate_median.median(axis=1)

df_aggregate_then_rank_median_ranking = rank_based_on_score(df_rank_then_aggregate_median['median'], teams)
df_aggregate_then_rank_median_ranking['Ranking Method'] = ['median-then-rank']*len(teams)
df_ranking_method.append(df_aggregate_then_rank_median_ranking)

df_ranking_method = pd.concat(df_ranking_method,0).reset_index()

sns.set_style("darkgrid")
fig, ax = plt.subplots(1, 1,figsize=(7,7))
g = sns.lineplot(x="Ranking Method", 
                 y="Rank", 
                 hue="Team",
                 data=df_ranking_method,
                 palette=color_ranked_team, 
                 hue_order=ranked_teams['CumRank_All']).invert_yaxis()
plt.yticks(list(range(1,len(teams)+1)))
plt.legend(bbox_to_anchor=(1.0, 0.9), loc=2, borderaxespad=0.)
fig.savefig('imgs2023_lnq/comparision_ranking.pdf', bbox_inches = 'tight', pad_inches = 0, dpi=100)               
                
                
# 7/ Boostrapping to test model robustness
print('Starting bootstrapping')
df_kendall = {'Kendall score':[],
              'Metric(s)':[]
              }

df_btstrp_final = {'Team':[],
                        'Rank':[],
                        'Metric(s)':[]}

method_names = {'CumRank_All':'Dice and ASSD', 
                'CumRank_Dice':' Dice', 
                'CumRank_ASSD':'ASSD'}

average_size = []
np.random.seed(1)
for b in tqdm(range(1000)): # 1000 bootstrap samples
    btstrp_cases = np.random.choice(cases, size=len(cases), replace=True)
    average_size.append(len(set(btstrp_cases))/len(cases))
    
    df_btstrp = []
    for c in btstrp_cases:
        df_btstrp_c = df_cum_rank[df_cum_rank['Case']==c]
        df_btstrp.append(df_btstrp_c)
    df_btstrp = pd.concat(df_btstrp).reset_index()
    
    assert(df_btstrp.shape[0]==len(cases)*len(teams))
    
    for method in ['CumRank_All', 'CumRank_Dice', 'CumRank_ASSD']:
        btstrp_ranking =  [
            df_btstrp[df_btstrp['Team']==t][method].mean() 
                for t in ranked_teams[method]
            ]
        btstrp_ranking = rankdata(btstrp_ranking, method='min')
        
        # Correlation with the final ranking using Kendall's tau
        tau = kendalltau(ranks_ranked_team, btstrp_ranking)[0]
        df_kendall['Kendall score'].append(tau)
        df_kendall['Metric(s)'].append(method_names[method])
        
        # Tracking rankings for each team
        for i,team in enumerate(ranked_teams[method]): 
            df_btstrp_final['Team'].append(team)
            df_btstrp_final['Metric(s)'].append(method_names[method])
            df_btstrp_final['Rank'].append(btstrp_ranking[i])
                  
df_kendall = pd.DataFrame(df_kendall)
df_btstrp_final = pd.DataFrame(df_btstrp_final)

print('Average proportion of unique elements in bootstraping: {:.4f}'.format(np.mean(average_size)))


# 8/ Comparision of the ranking robustness per set of metrics
fig = plt.figure()
ax = sns.violinplot(
    y="Kendall score", 
    x="Metric(s)", 
    data=df_kendall)

ax.set(
    ylabel="Kendall's τ",
)               
    

ax.set(ylim=(0.9, 1))
plt.title('Stability of the ranking scheme w.r.t the metric(s)')
plt.grid()
plt.show()
fig.savefig('imgs2023_lnq/stability_metrics.pdf', bbox_inches = 'tight', pad_inches = 0, dpi=100)



# 9/ Variability of achieved rankings across tasks
mask_btstrp_all = df_btstrp_final['Metric(s)']==method_names['CumRank_All']
df_btstrp_final_all = df_btstrp_final[mask_btstrp_all]

df_rank_variability = {'Team':[],
                       'Rank':[],
                       'Percentage':[]
                       }

for team in ranked_teams['CumRank_All']:
    mask_team = df_btstrp_final_all['Team']==team
    df_per_team = df_btstrp_final_all[mask_team]['Rank'].value_counts()
    df_per_team = df_per_team / 10
    for rank, per in df_per_team.iteritems():
        df_rank_variability['Team'].append(team)
        df_rank_variability['Rank'].append(rank)
        df_rank_variability['Percentage'].append(per)
        
    
df_rank_variability = pd.DataFrame(df_rank_variability)   


fig = plt.figure()    
g = sns.scatterplot(x="Rank", 
                    y="Team", 
                    hue="Team",
                    size="Percentage",
                    data=df_rank_variability,
                    palette=color_ranked_team, 
                    hue_order=ranked_teams['CumRank_All'])

h,l = g.get_legend_handles_labels()
box = g.get_position()
g.set_position([1.3*box.x0, box.y0, box.width * 0.85, box.height]) # resize position

plt.title('Stability of the ranking scheme using bootstrapping')
# Put a legend to the right side
g.legend(h[16+2:],[k+"%" for k in l[16+2:]],loc='right', bbox_to_anchor=(1.22, 0.5), ncol=1)

plt.show()    
fig.savefig('imgs2023_lnq/stability_bootstrapping.pdf', bbox_inches = 'tight', pad_inches = 0, dpi=100) 
    


    
    
    
