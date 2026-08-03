#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

TASKS=['religion','salary','politics','education']
TASK_LABELS={'religion':'Religion','salary':'Salary','politics':'Politics','education':'Education'}
LANGS=['en','es','zh']
LANG_LABELS={'en':'EN','es':'ES','zh':'ZH'}
PAIRS=[('en','es'),('en','zh'),('es','zh')]


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--output_dir',default='appendix_tables')
    p.add_argument('--decimals',type=int,default=2)
    p.add_argument('--p_decimals',type=int,default=3)
    p.add_argument('--partition',default='intersection')
    return p.parse_args()


def read_manifest(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,list) or not data:
        raise ValueError('Manifest must be a non-empty JSON list')
    return data


def load_all(manifest_path):
    names=[
        'rank_gaps_macro.csv',
        'cross_lingual_distribution_shift_macro.csv',
        'cross_lingual_pairwise_consistency.csv',
        'cross_lingual_agreement_bootstrap_ci.csv',
        'stuart_maxwell_tests.csv',
        'visual_text_ordinal_rank_differences.csv',
        'visual_text_paired_permutation_ordinal.csv',
        'language_demographic_cramers_v.csv',
    ]
    meta=[]; buckets={}
    for item in read_manifest(manifest_path):
        base=Path(item['metrics_dir'])
        if not base.is_absolute(): base=(Path(manifest_path).parent/base).resolve()
        model=str(item['model_name']); display=str(item.get('display_name',model)); size=str(item.get('size','')); order=int(item.get('order',9999))
        meta.append({'model_name':model,'display_name':display,'size':size,'model_order':order})
        for name in names:
            f=base/name
            if not f.exists(): continue
            df=pd.read_csv(f)
            df['model_name']=model; df['display_name']=display; df['size']=size; df['model_order']=order
            buckets.setdefault(name,[]).append(df)
    metadata=pd.DataFrame(meta).drop_duplicates('model_name').sort_values(['model_order','display_name']).reset_index(drop=True)
    merged={k:pd.concat(v,ignore_index=True) for k,v in buckets.items()}
    return metadata, merged


def esc(x): return str(x).replace('_',r'\_')
def fmt(x,d=2):
    try: x=float(x)
    except: return '--'
    return '--' if math.isnan(x) else f'{x:.{d}f}'
def fmtp(x,d=3):
    try: x=float(x)
    except: return '--'
    if math.isnan(x): return '--'
    t=10**(-d)
    return f'$<{t:.{d}f}$' if x<t else f'{x:.{d}f}'

def first(df, filters, col):
    if df is None or df.empty or col not in df.columns: return float('nan')
    z=df
    for k,v in filters.items():
        if k not in z.columns: return float('nan')
        z=z[z[k].astype(str).str.lower().eq(str(v).lower())]
    return float('nan') if z.empty else float(z.iloc[0][col])

def save(lines,path): Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')


def rank_gaps(meta,df,out,d,partition):
    L=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lllcccc}',r'\toprule',r'\textbf{Model} & \textbf{Size} & \textbf{Lang.} & \multicolumn{2}{c}{\textbf{Vision--language}} & \multicolumn{2}{c}{\textbf{Text-only}} \\',r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',r'& & & \textbf{Salary} & \textbf{Education} & \textbf{Salary} & \textbf{Education} \\',r'\midrule']
    for mi,m in meta.iterrows():
        for li,lang in enumerate(LANGS):
            mc=rf'\multirow{{3}}{{*}}{{{esc(m.display_name)}}}' if li==0 else ''
            sc=rf'\multirow{{3}}{{*}}{{{esc(m.size)}}}' if li==0 else ''
            vals=[]
            for cond in ['visual','text_only']:
                for task in ['salary','education']:
                    vals.append(first(df,{'model_name':m.model_name,'language':lang,'condition':cond,'task':task,'partition':partition},'mean_max_rank_gap'))
            L.append(f'{mc} & {sc} & {LANG_LABELS[lang]} & '+ ' & '.join(fmt(v,d) for v in vals)+r' \\')
        if mi<len(meta)-1:L.append(r'\addlinespace')
    L += [r'\bottomrule',r'\end{tabular}',r'\caption{Maximum demographic-group mean-rank gaps for salary and education. Scores are computed within occupations and macro-averaged across occupations.}',r'\label{tab:appendix_rank_gaps}',r'\end{table*}']
    save(L,out)


def lang_shift(meta,df,out,d):
    L=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lllcccccccccccc}',r'\toprule',r'& & & \multicolumn{4}{c}{\textbf{EN--ES}} & \multicolumn{4}{c}{\textbf{EN--ZH}} & \multicolumn{4}{c}{\textbf{ES--ZH}} \\',r'\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(lr){12-15}',r'\textbf{Model} & \textbf{Size} & \textbf{Cond.} & \textbf{Rel.} & \textbf{Sal.} & \textbf{Pol.} & \textbf{Edu.} & \textbf{Rel.} & \textbf{Sal.} & \textbf{Pol.} & \textbf{Edu.} & \textbf{Rel.} & \textbf{Sal.} & \textbf{Pol.} & \textbf{Edu.} \\',r'\midrule']
    for mi,m in meta.iterrows():
        for ci,cond in enumerate(['visual','text_only']):
            mc=rf'\multirow{{2}}{{*}}{{{esc(m.display_name)}}}' if ci==0 else ''
            sc=rf'\multirow{{2}}{{*}}{{{esc(m.size)}}}' if ci==0 else ''
            vals=[]
            for a,b in PAIRS:
                for task in TASKS:
                    vals.append(first(df,{'model_name':m.model_name,'condition':cond,'task':task,'language_1':a,'language_2':b},'mean_language_distribution_njsd'))
            L.append(f'{mc} & {sc} & {"Visual" if cond=="visual" else "Text-only"} & '+' & '.join(fmt(v,d) for v in vals)+r' \\')
        if mi<len(meta)-1:L.append(r'\addlinespace')
    L += [r'\bottomrule',r'\end{tabular}',r'}',r'\caption{Pairwise cross-lingual answer-distribution shift measured using nJSD and macro-averaged across occupations and demographic groups.}',r'\label{tab:appendix_language_shift}',r'\end{table*}']
    save(L,out)


def agreement_ci(meta,cons,ci,out):
    L=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{lllccc}',r'\toprule',r'\textbf{Model} & \textbf{Cond.} & \textbf{Task} & \textbf{EN--ES} & \textbf{EN--ZH} & \textbf{ES--ZH} \\',r'\midrule']
    for mi,m in meta.iterrows():
        firstrow=True
        for cond in ['visual','text_only']:
            for ti,task in enumerate(TASKS):
                mc=rf'\multirow{{8}}{{*}}{{{esc(m.display_name)}}}' if firstrow else ''; firstrow=False
                cc=rf'\multirow{{4}}{{*}}{{{"Visual" if cond=="visual" else "Text-only"}}}' if ti==0 else ''
                cells=[]
                for a,b in PAIRS:
                    ag=first(cons,{'model_name':m.model_name,'condition':cond,'task':task,'language_1':a,'language_2':b},'exact_agreement')
                    lo=first(ci,{'model_name':m.model_name,'condition':cond,'task':task,'language_1':a,'language_2':b},'agreement_ci_low')
                    hi=first(ci,{'model_name':m.model_name,'condition':cond,'task':task,'language_1':a,'language_2':b},'agreement_ci_high')
                    cells.append('--' if any(math.isnan(x) for x in [ag,lo,hi]) else f'{100*ag:.1f} [{100*lo:.1f}, {100*hi:.1f}]')
                L.append(f'{mc} & {cc} & {TASK_LABELS[task]} & '+' & '.join(cells)+r' \\')
            L.append(r'\cmidrule(lr){2-6}')
        L.pop()
        if mi<len(meta)-1:L.append(r'\addlinespace')
    L += [r'\bottomrule',r'\end{tabular}',r'\caption{Pairwise exact agreement with stratified-bootstrap 95\% confidence intervals. Values are percentages.}',r'\label{tab:appendix_agreement_ci}',r'\end{table*}']
    save(L,out)


def stuart(meta,df,out,pdigs):
    L=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lllcccccc}',r'\toprule',r'& & & \multicolumn{2}{c}{\textbf{EN--ES}} & \multicolumn{2}{c}{\textbf{EN--ZH}} & \multicolumn{2}{c}{\textbf{ES--ZH}} \\',r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',r'\textbf{Model} & \textbf{Cond.} & \textbf{Task} & $\chi^2$ & $p_{\mathrm{Holm}}$ & $\chi^2$ & $p_{\mathrm{Holm}}$ & $\chi^2$ & $p_{\mathrm{Holm}}$ \\',r'\midrule']
    for mi,m in meta.iterrows():
        firstrow=True
        for cond in ['visual','text_only']:
            for ti,task in enumerate(TASKS):
                mc=rf'\multirow{{8}}{{*}}{{{esc(m.display_name)}}}' if firstrow else ''; firstrow=False
                cc=rf'\multirow{{4}}{{*}}{{{"Visual" if cond=="visual" else "Text-only"}}}' if ti==0 else ''
                vals=[]
                for a,b in PAIRS:
                    vals += [fmt(first(df,{'model_name':m.model_name,'condition':cond,'task':task,'language_1':a,'language_2':b},'stuart_maxwell_statistic'),2), fmtp(first(df,{'model_name':m.model_name,'condition':cond,'task':task,'language_1':a,'language_2':b},'p_holm'),pdigs)]
                L.append(f'{mc} & {cc} & {TASK_LABELS[task]} & '+' & '.join(vals)+r' \\')
            L.append(r'\cmidrule(lr){2-9}')
        L.pop()
        if mi<len(meta)-1:L.append(r'\addlinespace')
    L += [r'\bottomrule',r'\end{tabular}',r'}',r'\caption{Stuart--Maxwell tests of marginal homogeneity for paired multilingual predictions. Holm-adjusted two-sided $p$-values are reported.}',r'\label{tab:appendix_stuart_maxwell}',r'\end{table*}']
    save(L,out)


def ordinal_modality(meta,diff,perm,out,d,pdigs):
    L=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{lllcccc}',r'\toprule',r'\textbf{Model} & \textbf{Size} & \textbf{Lang.} & \multicolumn{2}{c}{\textbf{Mean rank difference}} & \multicolumn{2}{c}{\textbf{Permutation $p_{\mathrm{Holm}}$}} \\',r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',r'& & & \textbf{Salary} & \textbf{Education} & \textbf{Salary} & \textbf{Education} \\',r'\midrule']
    for mi,m in meta.iterrows():
        for li,lang in enumerate(LANGS):
            mc=rf'\multirow{{3}}{{*}}{{{esc(m.display_name)}}}' if li==0 else ''
            sc=rf'\multirow{{3}}{{*}}{{{esc(m.size)}}}' if li==0 else ''
            ds=[]; ps=[]
            for task in ['salary','education']:
                z=diff[(diff['model_name'].astype(str)==str(m.model_name)) & (diff['language'].astype(str).str.lower()==lang) & (diff['task'].astype(str).str.lower()==task)] if diff is not None and not diff.empty else pd.DataFrame()
                ds.append(z['delta_mean_rank_visual_minus_text'].mean() if not z.empty and 'delta_mean_rank_visual_minus_text' in z.columns else np.nan)
                ps.append(first(perm,{'model_name':m.model_name,'language':lang,'task':task},'p_holm'))
            L.append(f'{mc} & {sc} & {LANG_LABELS[lang]} & {fmt(ds[0],d)} & {fmt(ds[1],d)} & {fmtp(ps[0],pdigs)} & {fmtp(ps[1],pdigs)} \\')
        if mi<len(meta)-1:L.append(r'\addlinespace')
    L += [r'\bottomrule',r'\end{tabular}',r'\caption{Visual-minus-text-only mean predicted-rank differences for salary and education, with paired permutation tests. Positive values indicate higher predicted ranks in the vision--language condition.}',r'\label{tab:appendix_ordinal_modality}',r'\end{table*}']
    save(L,out)


def cramer(meta,df,out,d,pdigs,partition):
    L=[r'\begin{table*}[t]',r'\centering',r'\small',r'\setlength{\tabcolsep}{3.5pt}',r'\begin{tabular}{lllcccccccc}',r'\toprule',r'& & & \multicolumn{4}{c}{\textbf{Cramér\'s $V$}} & \multicolumn{4}{c}{\textbf{$p$-value}} \\',r'\cmidrule(lr){4-7}\cmidrule(lr){8-11}',r'\textbf{Model} & \textbf{Size} & \textbf{Cond.} & \textbf{Rel.} & \textbf{Sal.} & \textbf{Pol.} & \textbf{Edu.} & \textbf{Rel.} & \textbf{Sal.} & \textbf{Pol.} & \textbf{Edu.} \\',r'\midrule']
    for mi,m in meta.iterrows():
        for ci,cond in enumerate(['visual','text_only']):
            mc=rf'\multirow{{2}}{{*}}{{{esc(m.display_name)}}}' if ci==0 else ''
            sc=rf'\multirow{{2}}{{*}}{{{esc(m.size)}}}' if ci==0 else ''
            vs=[]; ps=[]
            for task in TASKS:
                f={'model_name':m.model_name,'condition':cond,'task':task,'partition':partition}
                vs.append(first(df,f,'cramers_v')); ps.append(first(df,f,'chi_square_p'))
            L.append(f'{mc} & {sc} & {"Visual" if cond=="visual" else "Text-only"} & '+' & '.join(fmt(v,d) for v in vs)+' & '+' & '.join(fmtp(v,pdigs) for v in ps)+r' \\')
        if mi<len(meta)-1:L.append(r'\addlinespace')
    L += [r'\bottomrule',r'\end{tabular}',r'\caption{Associations among prompt language, intersectional demographic group, and selected answer, quantified using Cramér\'s $V$.}',r'\label{tab:appendix_cramers_v}',r'\end{table*}']
    save(L,out)


def main():
    a=args(); manifest=Path(a.manifest).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    meta,M=load_all(manifest)
    rank_gaps(meta,M.get('rank_gaps_macro.csv',pd.DataFrame()),out/'appendix_ordinal_rank_gaps.tex',a.decimals,a.partition)
    lang_shift(meta,M.get('cross_lingual_distribution_shift_macro.csv',pd.DataFrame()),out/'appendix_cross_lingual_distribution_shift.tex',a.decimals)
    agreement_ci(meta,M.get('cross_lingual_pairwise_consistency.csv',pd.DataFrame()),M.get('cross_lingual_agreement_bootstrap_ci.csv',pd.DataFrame()),out/'appendix_agreement_bootstrap_ci.tex')
    stuart(meta,M.get('stuart_maxwell_tests.csv',pd.DataFrame()),out/'appendix_stuart_maxwell.tex',a.p_decimals)
    ordinal_modality(meta,M.get('visual_text_ordinal_rank_differences.csv',pd.DataFrame()),M.get('visual_text_paired_permutation_ordinal.csv',pd.DataFrame()),out/'appendix_visual_text_ordinal.tex',a.decimals,a.p_decimals)
    cramer(meta,M.get('language_demographic_cramers_v.csv',pd.DataFrame()),out/'appendix_cramers_v.tex',a.decimals,a.p_decimals,a.partition)
    master=[r'\section{Additional Metric Results}',r'\label{appendix:additional_metrics}',r'\subsection{Ordinal Demographic Rank Gaps}',r'\input{appendix_tables/appendix_ordinal_rank_gaps}',r'\subsection{Cross-Lingual Distribution Shift}',r'\input{appendix_tables/appendix_cross_lingual_distribution_shift}',r'\subsection{Bootstrap Confidence Intervals}',r'\input{appendix_tables/appendix_agreement_bootstrap_ci}',r'\subsection{Marginal-Homogeneity Tests}',r'\input{appendix_tables/appendix_stuart_maxwell}',r'\subsection{Ordinal Vision--Language and Text-Only Comparison}',r'\input{appendix_tables/appendix_visual_text_ordinal}',r'\subsection{Language--Demographic Associations}',r'\input{appendix_tables/appendix_cramers_v}']
    save(master,out/'appendix_additional_metrics.tex')
    print(out)

if __name__=='__main__': main()
