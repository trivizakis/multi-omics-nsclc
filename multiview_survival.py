#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: trivizakis

@github: github.com/trivizakis
"""
import os
import sys
import random
import pandas as pd
import numpy as np
import pickle as pkl

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

from imblearn.over_sampling import SMOTE

from nsclc_objectives import objective_maker

from sklearn.preprocessing import robust_scale

from sklearn.model_selection import StratifiedKFold

from matplotlib import pyplot as plt

from sklearn.feature_selection import f_classif as fc
from sklearn.feature_selection import SelectKBest as kbest
from sklearn.feature_selection import VarianceThreshold as va

from numpy import interp
import math
import scipy.stats as st

from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, Matern

from sksurv.ensemble import RandomSurvivalForest, ExtraSurvivalTrees
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.tree import SurvivalTree
from sksurv.svm import NaiveSurvivalSVM, HingeLossSurvivalSVM, FastKernelSurvivalSVM, FastSurvivalSVM, MinlipSurvivalAnalysis

from sksurv.metrics import concordance_index_censored

def get_ml_model(classifier):
    if classifier == "Forest-surv":
        ml_model = RandomSurvivalForest(n_estimators=1000,
                                                       min_samples_split=6,
                                                       min_samples_leaf=3,
                                                       max_features="sqrt",
                                                       n_jobs=5)
    if classifier == "Surv-Tree":
        ml_model = SurvivalTree()
    elif classifier == "Extra-Tree":
        ml_model = ExtraSurvivalTrees()
    elif classifier == "Cox":
        ml_model = CoxnetSurvivalAnalysis(l1_ratio=0.99, fit_baseline_model=True)
    elif classifier == "CoxPH":
        ml_model = CoxnetSurvivalAnalysis(l1_ratio=0.99, fit_baseline_model=True)
    elif classifier == "Hinge-SVM":
        ml_model = HingeLossSurvivalSVM()
    elif classifier == "FastKernel-SVM":
        ml_model = FastKernelSurvivalSVM()
    elif classifier == "Fast-SVM":
        ml_model = FastSurvivalSVM()
    elif classifier == "Min-SVM":
        ml_model = MinlipSurvivalAnalysis()
    elif classifier == "Naive-SVM":
        ml_model = NaiveSurvivalSVM()
    return ml_model
                    
def get_normalized_value(value,mean,std):
    value = (value-mean)/std
    if value>1:
        value=1
    elif value<-1:
        value=-1
    return value

def get_normalized_value_categorical(value):
    if value==0:
        return -1
    else:
        return value
    
def get_composite_score(feature_vector,coefs):
    score=0
    for index, feature in enumerate(feature_vector):
        score+=feature*coefs[index]
    return score

def get_composite_dict(dicts, coefs):
    #dicts: list of dictionaries (patterns)
    #coefs: list of coefs for each pattern
    score_dict={}
    
    for key in list(dicts[0].keys()):
        combined_score=0
        scores = []
        for index in range(0,len(dicts)):
            score = get_composite_score(dicts[index][key],coefs[index])
            scores.append(score)
            combined_score += score
        scores.append(combined_score)
        score_dict[key]= np.array(scores).reshape(-1,)
    return score_dict
            

def plot_survival(model, CI, path, X, y, clf):
    
    if clf == "Hinge-SVM" or "FastKernel-SVM" or "Fast-SVM" or "Min-SVM" or "Naive-SVM":
        prediction = model.predict(X)
        results = concordance_index_censored(y['Status'].astype(np.bool), y['Survival_in_days'].values, -prediction)
        print("CI: "+str(round(results[0], 3)))
        print("Concordant: "+str(results[1]))
        print("Discordant: "+str(results[2]))
        print("Tied_risk: "+str(results[3]))
        print("Tied_time: "+str(results[4]))
        
        file2write=open(path+"performance_CI",'w')
        file2write.write("CI: "+str(round(results[0], 3))+str("\n"))
        file2write.write("Concordant: "+str(results[1])+str("\n"))
        file2write.write("Discordant: "+str(results[2])+str("\n"))
        file2write.write("Tied_risk: "+str(results[3])+str("\n"))
        file2write.write("Tied_time: "+str(results[4])+str("\n"))
        file2write.close()
    else:
        if clf == "Forest-surv":
            surv = model.predict_survival_function(X, return_array=True)
            for i, s in enumerate(surv):
                plt.step(model.event_times_, s, where="post", label=str(i))
        elif clf == "Extra-Tree":
            surv = model.predict_survival_function(X, return_array=True)
            for i, s in enumerate(surv):
                plt.step(model.event_times_, s, where="post", label=str(i))
        elif clf == "Cox":
            surv = model.predict_survival_function(X)
            surv_funcs = {}
            for alpha in model.alphas_[:5]:
                surv_funcs[alpha] = model.predict_survival_function(X, alpha=alpha)#X.iloc[:1]
            for alpha, surv_alpha in surv_funcs.items():
                for fn in surv_alpha:
                    plt.step(fn.x, fn(fn.x), where="post",
                         label="alpha = {:.3f}".format(alpha))
        elif clf =="CoxPH":
            surv = model.predict_survival_function(X)
            surv_funcs = {}
            for alpha in model.alphas_[:5]:
                surv_funcs[alpha] = model.predict_survival_function(X, alpha=alpha)#X.iloc[:1]
            for alpha, surv_alpha in surv_funcs.items():
                for fn in surv_alpha:
                    plt.step(fn.x, fn(fn.x), where="post",
                         label="alpha = {:.3f}".format(alpha))
        elif clf =="Surv-Tree":
            surv = model.predict_survival_function(X, return_array=True)
            for i, s in enumerate(surv):
                plt.step(model.event_times_, s, where="post", label=str(i))
        plt.ylabel("Survival probability")
        plt.xlabel("Time in days")
        plt.legend(X.index)
        plt.title("CI %.4f" % CI)
        plt.grid(True)
        plt.savefig(path+"survival analysis.png", dpi=300, bbox_inches="tight")
        plt.clf()
    
def apply_feature_selection(df, labels, cutoff_pvalue=0.05, c_value=1):
    X=[]
    
    for key in list(df.index):
        X.append(df.loc[key])
    X = np.array(X)
    y = np.hstack(labels)
    
    remover = va(threshold=0.2)
    X = remover.fit_transform(X)
    # variance_mask = remover.get_support()
    variance_indices = remover.get_support(indices=True)
    # print(variance_indices)
    # print(np.unique(variance_mask,return_counts=True)) #indices=True
    
    f_scores, p_values = fc(X, y)
    critical_value = st.f.ppf(q=1-cutoff_pvalue, dfn=len(np.unique(y))-1, dfd=len(y)-len(np.unique(y)))
    
    best_indices=[]
    for index, p_value in enumerate(p_values):
        if f_scores[index]>=critical_value and p_value<cutoff_pvalue:
            best_indices.append(index)
    print("Best ANOVA features: "+str(len(best_indices)))
    
    df = df.iloc[:,variance_indices]
    # print(df.shape)
    best_columns = np.array(list(df.columns))[best_indices]
    best_features = np.array(list(df[best_columns].values))
    
    sel_ = SelectFromModel(LogisticRegression(C=c_value, penalty='l1', solver="liblinear"))
    # sel_ = SelectFromModel(LinearSVR(C=0.1))
    sel_.fit(best_features, y)
    selected_features_bool = sel_.get_support()    
    coef = sel_.estimator_.coef_.reshape(-1,)
    
    final_selected=[]
    coefs = []
    if len(selected_features_bool)<1:
        final_selected.append(np.array(list(df.columns))[best_indices[0]])
        coefs.append(1)
    else:
        for index,feat_id in enumerate(best_columns):
            if selected_features_bool[index]:
                final_selected.append(feat_id)
                coefs.append(coef[index])
        print("Best l1 features: "+str(len(final_selected)))
    
    return np.array(final_selected),np.array(coefs)

#Parameters
hypes={}
hypes["dataset_dir"] = "path/to/dataset/"

#load transcriptomics
transcriptomics = pd.read_excel(hypes["dataset_dir"]+"RNAseq.xls")

#load radiomics
radiomics = pd.read_excel(hypes["dataset_dir"]+"radiomics.xls")

#load deep feratures
deep = pd.read_excel(hypes["dataset_dir"]+"deep_features_raw.xls")

#load clinical data
clinical=pd.read_excel(hypes["dataset_dir"]+"clinical_data.xls")
clinical_data=pd.read_excel(hypes["dataset_dir"]+"clinical_data_days.xls")

#load labels
labels=pd.read_excel(hypes["dataset_dir"]+"labels.xls")

# Analysis
exps = ["SURVIVAL-MEDIAN-DAYS"]#"SURVIVAL-OVERALL-DAYS"

results={}
r_results={}
d_results={}
t_results={}
c_results={}
tcs_results={}
rs_results={}
rts_results={}
ds_results={}
mvs_results={}
failed=[]
for classifier in ["Surv-Tree","Extra-Tree","CoxPH","Cox","Forest-surv","Hinge-SVM","FastKernel-SVM","Fast-SVM","Min-SVM","Naive-SVM"]:
    results_path="results_"+classifier
    os.mkdir("chkp/"+results_path)
    for model_name in list(deep.keys()):
        for experiment in exps:#survival type
            for expindex in range(0,1):
                splits={}
                pids = np.array(list(labels[experiment].keys()),dtype=str)
                # set_ = np.array(list(dataset_.values()))
                f_labels = np.array(list(labels[experiment].values()))
                sss = StratifiedKFold(n_splits=4, shuffle=True)          
                kfolds=[]          
                for train_index, test_index in sss.split(pids, f_labels):
                    kfolds.append([pids[train_index], pids[test_index]])
                splits[experiment] = kfolds

                Concordance_index = []
                CI_deep = []
                CI_radiomics = []
                CI_transcriptomics = []
                CI_clinical = []
                
                CI_transcriptomics_score=[]
                CI_radiomics_score=[]
                CI_radiotranscriptomics_score=[]
                CI_deep_score=[]
                CI_multiview_score=[]
                for index,split in enumerate(splits[experiment]):                    
                    # print("feature selection, deep: "+model_name+" n_feat: "+str(n_feat)+" clf: "+classifier+" exp: "+experiment)
                    g_tr_split=[]
                    g_tst_split=[]
                    for key in list(transcriptomics.index):
                        if key in list(split[0]):
                            g_tr_split.append(key)
                        elif key in list(split[1]):
                            g_tst_split.append(key)
                            
                    r_tr_split=[]
                    r_tst_split=[]
                    for key in list(radiomics.index):
                        if key in list(split[0]):
                            r_tr_split.append(key)
                        elif key in list(split[1]):
                            r_tst_split.append(key)
                                        
                    deep_ = robust_scale(deep[model_name],unit_variance=True,quantile_range=(10.0, 90.0))
                    deep_[deep_>1]=1
                    deep_[deep_<-1]=-1
                    
                    columns_=["f"+str(s) for s in deep[model_name].columns]
                    deep_ = pd.DataFrame(data=deep_,index=deep[model_name].index, columns=columns_)#, columns=deep[model_name].columns)
                    
                    #labels for single source experiments
                    g_labels=[]
                    for pid in list(transcriptomics.loc[g_tr_split].index):
                        try:
                            g_labels.append(labels[experiment][pid])
                        except:
                            continue
                    
                    r_labels=[]
                    for pid in list(radiomics.loc[r_tr_split].index):
                        try:
                            r_labels.append(labels[experiment][pid])
                        except:
                            continue
                        
                    d_labels=[]
                    for pid in list(deep_.loc[r_tr_split].index):
                        try:
                            d_labels.append(labels[experiment][pid])
                        except:
                            continue
                        
                    #feature selection
                    print("-feature selection, deep: "+model_name+" clf: "+classifier+" exp: "+experiment)                            
                    try:
                        print("---Transcriptomics")
                        transcriptomics_feat, transcriptomics_coefs = apply_feature_selection(transcriptomics.loc[g_tr_split], g_labels, cutoff_pvalue=0.005, c_value=0.6)
                        print("---Radiomics")
                        radiomic_feat, radiomic_coefs = apply_feature_selection(radiomics.loc[r_tr_split], r_labels, cutoff_pvalue=0.05, c_value=0.8)
                        print("---Deep")
                        deep_feat, deep_coefs = apply_feature_selection(deep_.loc[r_tr_split], r_labels, cutoff_pvalue=0.05, c_value=0.6)
                        print("---Clinical")
                        # clinic_feat, clinic_coefs = apply_feature_selection(clinical.loc[r_tr_split], r_labels, cutoff_pvalue=0.05, c_value=1)
                        clinic_feat = ["AGE","GENDER","PACK-YEARS","SMOKING","WEIGHT","ADJUVANT","CHEMO","RADIO","ETHNICITY","HISTO-GRADE","PLEURAL","GG","LOCATION","MERGED-T-STAGE"]
                        clinic_coefs= [0.05,0.05,0.10,0.10,0.05,0.20,0.20,0.20,0.05,0.10,0.20,0.20,0.20,0.20]
                    except:
                        failed = experiment+"_"+model_name+"_"+"_"+classifier
                        continue                    
                    
                    path = "chkp/"+results_path+"/"+experiment+"_"+model_name+"_"+classifier+"_nsp"+str(index+1)+"_expindex"+str(expindex)
                    os.mkdir(path)
                    np.save(path+"/selected_gen",np.array(transcriptomics_feat,dtype=str))
                    np.save(path+"/selected_rad",np.array(radiomic_feat,dtype=str))
                    np.save(path+"/selected_deep",np.array(deep_feat,dtype=str))
                    np.save(path+"/selected_clinical",np.array(clinic_feat,dtype=str))
                    np.save(path+"/selected_gen_coefs",np.array(transcriptomics_coefs,dtype=str))
                    np.save(path+"/selected_rad_coefs",np.array(radiomic_coefs,dtype=str))
                    np.save(path+"/selected_deep_coefs",np.array(deep_coefs,dtype=str))
                    np.save(path+"/selected_clinical_coefs",np.array(clinic_coefs,dtype=str))
                    np.save(path+"/tr_split",np.array(r_tr_split,dtype=str))
                    np.save(path+"/tst_split",np.array(r_tst_split,dtype=str)) 
                    
                    selected_transcriptomics={}
                    for key in list(transcriptomics.index):
                        selected_transcriptomics[key] = transcriptomics[transcriptomics_feat].loc[key].to_numpy()
                        
                    selected_radiomics={}
                    for key in list(radiomics.index):
                        selected_radiomics[key] = radiomics[radiomic_feat].loc[key].to_numpy()
                    
                    selected_deep={}
                    for key in list(deep_.index):
                        selected_deep[key] = deep_[deep_feat].loc[key].to_numpy()
                        
                    #check pattern availability per view    
                    # combined_patterns={}
                    # for key in list(selected_radiomics.keys()):
                    #     try:
                    #         combined_patterns[key] = np.concatenate((clinical[key],selected_transcriptomics[key],selected_radiomics[key],selected_deep[key]))
                    #     except:
                    #         # print(key)
                    #         continue
                        
                    # align multi-view labels
                    rg_labels={}
                    rg_patterns={}
                    for key in sorted(selected_radiomics.keys()):
                        try:
                            rg_labels[key] = labels[experiment][key]
                            # rg_patterns[key]=combined_patterns[key]
                            rg_patterns[key] = np.concatenate((clinical[key],selected_transcriptomics[key],selected_radiomics[key],selected_deep[key]))
                        except:
                            # print(key+" not labeled!")
                            continue
                            
                    r_labels={}
                    r_patterns={}
                    d_patterns={}
                    c_patterns={}
                    g_patterns={}
                    for key in sorted(rg_labels.keys()):
                        try:
                            r_labels[key]=labels[experiment][key]                        
                            
                            g_patterns[key]=selected_transcriptomics[key]
                            r_patterns[key]=selected_radiomics[key]
                            d_patterns[key]=selected_deep[key]
                            c_patterns[key]=clinical[key]
                        except:
                            #print(key+" not labeled!")
                            continue
                        
                    transcriptomics_score = get_composite_dict([g_patterns], [transcriptomics_coefs])
                    radiomics_score = get_composite_dict([r_patterns], [radiomic_coefs])
                    deep_score = get_composite_dict([d_patterns], [deep_coefs])
                    radiomics_scores = get_composite_dict([r_patterns,d_patterns], [radiomic_coefs,deep_coefs])
                    radiotranscriptomics_score = get_composite_dict([g_patterns,r_patterns,d_patterns], [transcriptomics_coefs, radiomic_coefs, deep_coefs])
                    multiview_score = get_composite_dict([g_patterns, r_patterns, d_patterns, c_patterns], [transcriptomics_coefs, radiomic_coefs, deep_coefs, clinic_coefs])
                                        
                    x_pd = pd.DataFrame.from_dict(rg_patterns, orient='index')  
                    x_pd_deep = pd.DataFrame.from_dict(d_patterns, orient='index')  
                    x_pd_clinical = pd.DataFrame.from_dict(c_patterns, orient='index')  
                    x_pd_radiomics = pd.DataFrame.from_dict(r_patterns, orient='index')  
                    x_pd_transcriptomics = pd.DataFrame.from_dict(g_patterns, orient='index')  
                    
                    x_pd_transcriptomics_score = pd.DataFrame.from_dict(transcriptomics_score, orient='index')  
                    x_pd_radiomics_score = pd.DataFrame.from_dict(radiomics_score, orient='index')  
                    x_pd_deep_score = pd.DataFrame.from_dict(deep_score, orient='index')  
                    x_pd_radiotranscriptomics_score = pd.DataFrame.from_dict(radiotranscriptomics_score, orient='index')  
                    x_pd_multiview_score = pd.DataFrame.from_dict(multiview_score, orient='index')  
                    
                    days_pd = pd.DataFrame.from_dict(clinical_data['SURVIVAL-MEDIAN-DAYS'], orient='index', columns=["Survival_in_days"])
                    status_pd = pd.DataFrame.from_dict(clinical_data['SURVIVAL-MEDIAN'], orient='index', columns=["Status"])
                    
                    y_pd = status_pd.join(days_pd, sort=True)#, on="Patient", how="left")
                    
                    X_train = x_pd.loc[r_tr_split]
                    X_test = x_pd.loc[r_tst_split]                    
                    X_train_deep = x_pd_deep.loc[r_tr_split]
                    X_test_deep = x_pd_deep.loc[r_tst_split]                    
                    X_train_clinical = x_pd_clinical.loc[r_tr_split]
                    X_test_clinical = x_pd_clinical.loc[r_tst_split]                    
                    X_train_radiomics = x_pd_radiomics.loc[r_tr_split]
                    X_test_radiomics = x_pd_radiomics.loc[r_tst_split]                    
                    X_train_transcriptomics = x_pd_transcriptomics.loc[r_tr_split]
                    X_test_transcriptomics = x_pd_transcriptomics.loc[r_tst_split]
                    
                    X_train_transcriptomics_score = x_pd_transcriptomics_score.loc[r_tr_split]
                    X_test_transcriptomics_score = x_pd_transcriptomics_score.loc[r_tst_split]                    
                    X_train_radiomics_score = x_pd_radiomics_score.loc[r_tr_split]
                    X_test_radiomics_score = x_pd_radiomics_score.loc[r_tst_split]                    
                    X_train_radiotranscriptomics_score = x_pd_radiotranscriptomics_score.loc[r_tr_split]
                    X_test_radiotranscriptomics_score = x_pd_radiotranscriptomics_score.loc[r_tst_split]                    
                    X_train_deep_score = x_pd_deep_score.loc[r_tr_split]
                    X_test_deep_score = x_pd_deep_score.loc[r_tst_split]                    
                    X_train_multiview_score = x_pd_multiview_score.loc[r_tr_split]
                    X_test_multiview_score = x_pd_multiview_score.loc[r_tst_split]
                    
                    y_train = y_pd.loc[r_tr_split]
                    y_test = y_pd.loc[r_tst_split]
                    
                    y_train_np = y_train.to_numpy(dtype=np.float)
                    y_test_np = y_test.to_numpy(dtype=np.float)                    
                   
                    print("classification: "+classifier)
                    struct_arr_train = y_train.astype({'Status':'?','Survival_in_days':'<f8'}).dtypes
                    y_train_np = np.array([tuple(x) for x in y_train.values], dtype=list(zip(struct_arr_train.index,struct_arr_train)))  
                    struct_arr_test = y_test.astype({'Status':'?','Survival_in_days':'<f8'}).dtypes
                    y_test_np = np.array([tuple(x) for x in y_test.values], dtype=list(zip(struct_arr_test.index,struct_arr_test)))  
                    # print(y_test_np)
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train, y_train_np)
                    ci = ml_model.score(X_test, y_test_np)                
                    Concordance_index.append(ci)
                    # plot_survival(ml_model, ci, path+"/multi-view_", X_test, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/multi-view_clr.pkl","wb"))    
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_deep, y_train_np)              
                    ci = ml_model.score(X_test_deep, y_test_np)
                    CI_deep.append(ci)
                    # plot_survival(ml_model, ci, path+"/deep_", X_test_deep, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/deep_clr.pkl","wb"))    
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_radiomics, y_train_np)   
                    ci = ml_model.score(X_test_radiomics, y_test_np)
                    CI_radiomics.append(ci)
                    # plot_survival(ml_model, ci, path+"/radiomics_", X_test_radiomics, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/radiomics_clr.pkl","wb"))    
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_transcriptomics, y_train_np)    
                    ci = ml_model.score(X_test_transcriptomics, y_test_np)
                    CI_transcriptomics.append(ci)
                    # plot_survival(ml_model, ci, path+"/transcriptomics_", X_test_transcriptomics, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/transcriptomics_clr.pkl","wb"))
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_clinical, y_train_np)    
                    ci =ml_model.score(X_test_clinical, y_test_np)
                    CI_clinical.append(ci)
                    # plot_survival(ml_model, ci, path+"/clinical_", X_test_clinical, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/clinical_clr.pkl","wb"))
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_transcriptomics_score, y_train_np)    
                    ci =ml_model.score(X_test_transcriptomics_score, y_test_np)
                    CI_transcriptomics_score.append(ci)
                    # plot_survival(ml_model, ci, path+"/transcriptomics_score_", X_test_transcriptomics_score, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/transcriptomics_score_clr.pkl","wb"))
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_radiomics_score, y_train_np)    
                    ci =ml_model.score(X_test_radiomics_score, y_test_np)
                    CI_radiomics_score.append(ci)
                    # plot_survival(ml_model, ci, path+"/radiomics_score_", X_test_radiomics_score, y_test, classifier)  
                    pkl.dump(ml_model, open(path+"/radiomics_score_clr.pkl","wb"))                    
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_radiotranscriptomics_score, y_train_np)    
                    ci =ml_model.score(X_test_radiotranscriptomics_score, y_test_np)
                    CI_radiotranscriptomics_score.append(ci)
                    # plot_survival(ml_model, ci, path+"/radiotranscriptomics_score_", X_test_radiotranscriptomics_score, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/radiotranscriptomics_score_clr.pkl","wb")) 
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_deep_score, y_train_np)    
                    ci =ml_model.score(X_test_deep_score, y_test_np)
                    CI_deep_score.append(ci)
                    # plot_survival(ml_model, ci, path+"/deep_score_", X_test_deep_score, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/deep_score_clr.pkl","wb"))
                    
                    ml_model = get_ml_model(classifier)
                    ml_model.fit(X_train_multiview_score, y_train_np)    
                    ci =ml_model.score(X_test_multiview_score, y_test_np)
                    CI_multiview_score.append(ci)
                    # plot_survival(ml_model, ci, path+"/multiview_score_", X_test_multiview_score, y_test, classifier)
                    pkl.dump(ml_model, open(path+"/multiview_score_clr.pkl","wb"))
                    
                print("____Multi-view results____")    
                print('List of possible CI:', Concordance_index)
                print('\nMaximum CI That can be obtained from this model is:',np.array(Concordance_index).max())
                print('\nMinimum CI:',np.array(Concordance_index).min())
                print('\nMean CI:',np.array(Concordance_index).mean())
                print('\nStandard Deviation is:', np.array(Concordance_index).std())
            
                results["multi-view: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(Concordance_index).max(),
                                                                          "Minimum CI":np.array(Concordance_index).min(),
                                                                          "Mean CI":np.array(Concordance_index).mean(),
                                                                          "Standard Deviation":np.array(Concordance_index).std()})   
                
                d_results["deep: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_deep).max(),
                                                                          "Minimum CI":np.array(CI_deep).min(),
                                                                          "Mean CI":np.array(CI_deep).mean(),
                                                                          "Standard Deviation":np.array(CI_deep).std()})  
                
                r_results["radiomics: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_radiomics).max(),
                                                                          "Minimum CI":np.array(CI_radiomics).min(),
                                                                          "Mean CI":np.array(CI_radiomics).mean(),
                                                                          "Standard Deviation":np.array(CI_radiomics).std()})  
                        
                t_results["transcriptomics: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_transcriptomics).max(),
                                                                          "Minimum CI":np.array(CI_transcriptomics).min(),
                                                                          "Mean CI":np.array(CI_transcriptomics).mean(),
                                                                          "Standard Deviation":np.array(CI_transcriptomics).std()})  
                        
                        
                c_results["clinical: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_clinical).max(),
                                                                          "Minimum CI":np.array(CI_clinical).min(),
                                                                          "Mean CI":np.array(CI_clinical).mean(),
                                                                          "Standard Deviation":np.array(CI_clinical).std()})  
                
                tcs_results["transcriptomics_score: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_transcriptomics_score).max(),
                                                                          "Minimum CI":np.array(CI_transcriptomics_score).min(),
                                                                          "Mean CI":np.array(CI_transcriptomics_score).mean(),
                                                                          "Standard Deviation":np.array(CI_transcriptomics_score).std()})
                
                rs_results["radiomics_score: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_radiomics_score).max(),
                                                                          "Minimum CI":np.array(CI_radiomics_score).min(),
                                                                          "Mean CI":np.array(CI_radiomics_score).mean(),
                                                                          "Standard Deviation":np.array(CI_radiomics_score).std()})
                
                rts_results["radiotranscriptomics_score: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_radiotranscriptomics_score).max(),
                                                                          "Minimum CI":np.array(CI_radiotranscriptomics_score).min(),
                                                                          "Mean CI":np.array(CI_radiotranscriptomics_score).mean(),
                                                                          "Standard Deviation":np.array(CI_radiotranscriptomics_score).std()})
                
                ds_results["deep_score: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_deep_score).max(),
                                                                          "Minimum CI":np.array(CI_deep_score).min(),
                                                                          "Mean CI":np.array(CI_deep_score).mean(),
                                                                          "Standard Deviation":np.array(CI_deep_score).std()})
                
                mvs_results["multi-view_score: "+model_name+", clf:"+ classifier+" "+str(expindex)] = pd.Series({"Maximum":np.array(CI_multiview_score).max(),
                                                                          "Minimum CI":np.array(CI_multiview_score).min(),
                                                                          "Mean CI":np.array(CI_multiview_score).mean(),
                                                                          "Standard Deviation":np.array(CI_multiview_score).std()})

                    
                    
            
multiview_results = pd.DataFrame.from_dict(results, orient="index")
r_results = pd.DataFrame.from_dict(r_results, orient="index")
d_results = pd.DataFrame.from_dict(d_results, orient="index")
t_results = pd.DataFrame.from_dict(t_results, orient="index")
c_results = pd.DataFrame.from_dict(c_results, orient="index")
tcs_results = pd.DataFrame.from_dict(tcs_results, orient="index")
rs_results = pd.DataFrame.from_dict(rs_results, orient="index")
rts_results = pd.DataFrame.from_dict(rts_results, orient="index")
ds_results = pd.DataFrame.from_dict(ds_results, orient="index")
mvs_results = pd.DataFrame.from_dict(mvs_results, orient="index")

multiview_results.to_csv("chkp/"+results_path+"/multi-view_results.csv")
r_results.to_csv("chkp/"+results_path+"/radiomics_results.csv")
d_results.to_csv("chkp/"+results_path+"/deep_results.csv")
t_results.to_csv("chkp/"+results_path+"/transcriptomics_results.csv")
c_results.to_csv("chkp/"+results_path+"/clinical_results.csv")
tcs_results.to_csv("chkp/"+results_path+"/tcs_results.csv")
rs_results.to_csv("chkp/"+results_path+"/rs_results.csv")
rts_results.to_csv("chkp/"+results_path+"/rts_results.csv")
ds_results.to_csv("chkp/"+results_path+"/ds_results.csv")
mvs_results.to_csv("chkp/"+results_path+"/mvs_results.csv")