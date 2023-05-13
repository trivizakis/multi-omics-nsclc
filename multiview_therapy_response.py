#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: trivizakis

@github: github.com/trivizakis
"""
import os
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import robust_scale
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.feature_selection import f_classif as fc
from sklearn.feature_selection import SelectKBest as kbest
from sklearn.feature_selection import VarianceThreshold as va
from numpy import interp
import scipy.stats as st
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix, f1_score
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared, Matern

def get_features(selected, df, selected_features):
    for key in list(df.index):
        selected[key] = df[selected_features].loc[key].to_numpy()
    return selected
    
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
                
def plot_roc(tprs, mean_auc, title):
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)
    
    base_fpr = np.linspace(0, 1, 101)
    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    
    plt.plot(base_fpr, mean_tprs, 'b', label="Mean AUC=%.2f" % mean_auc)
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title(title)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("chkp/figs/"+title+".tiff", dpi=300)
    # plt.show()

def machine_learning(dataset_, labels_, split, classifier, multiview=False):
    if not multiview:    
        train_pids = split[0]
        test_pids = split[1]
        
        train_set = []
        train_labels =[]
        for key in train_pids:
            try:
                train_set.append(dataset_[key])
                train_labels.append(labels_[key])
            except:
                print(key+" not available features IN TRAINING SET!")   
                continue             
        train_set = np.array(train_set)
        train_labels = np.stack(train_labels)
        
        oversample = SMOTE(sampling_strategy=1)
        train_set, train_labels = oversample.fit_resample(train_set, train_labels)
    else: # multiview
        c_train_pids = split["clinical"][0]
        c_test_pids = split["clinical"][1]
        d_train_pids = split["deep"][0]
        d_test_pids = split["deep"][1]
        r_train_pids = split["radiomics"][0]
        r_test_pids = split["radiomics"][1]
        g_train_pids = split["transcriptomics"][0]
        g_test_pids = split["transcriptomics"][1]
        
        d_train_set = []
        r_train_set = []
        g_train_set = []
        c_train_set = []
        train_labels=[]
        #align sets and labels
        for key in r_train_pids:
            try:
                train_labels.append(labels_[key])
                d_train_set.append(dataset_["deep"][key])
                r_train_set.append(dataset_["radiomics"][key])
                g_train_set.append(dataset_["transcriptomics"][key])
                c_train_set.append(dataset_["clinical"][key])
            except:
                # print(key+" not available features IN TRAINING SET!")   
                continue             
        d_train_set = np.array(d_train_set)
        r_train_set = np.array(r_train_set)
        g_train_set = np.array(g_train_set)
        c_train_set = np.array(c_train_set)
        train_labels = np.stack(train_labels)
                
        oversample = SMOTE(sampling_strategy=1)
        c_train_set, f_train_labels = oversample.fit_resample(c_train_set, train_labels)
        d_train_set, _ = oversample.fit_resample(d_train_set, train_labels)
        r_train_set, _ = oversample.fit_resample(r_train_set, train_labels)
        g_train_set, _ = oversample.fit_resample(g_train_set, train_labels)
        train_labels = f_train_labels
        
        #concatenate features
        train_set = np.concatenate((c_train_set, d_train_set, r_train_set, g_train_set),axis=1)
        #check testing set
        test_pids = list(set(r_test_pids) | set(g_test_pids) | set(d_test_pids) | set(c_test_pids) ) 
    test_set = []
    test_labels =[]
    #align testing set and testing labels
    for key in test_pids:
        try:
            test_labels.append(labels_[key])
            if multiview:
                c_pattern = dataset_["clinical"][key]
                d_pattern = dataset_["deep"][key]
                r_pattern = dataset_["radiomics"][key]
                g_pattern = dataset_["transcriptomics"][key]
                #concatenate features
                test_set.append(np.concatenate((c_pattern, d_pattern, r_pattern, g_pattern),axis=0))
            else:
                test_set.append(dataset_[key])
        except:
            # print(key+" not available features IN TESTING SET!")
            continue
    test_set = np.array(test_set)
    test_labels = np.stack(test_labels)
    
    if classifier == "poly_svm":
        clf = svm.SVC(kernel="poly", gamma="auto", probability=True)
    elif classifier == "linear_svm":
        clf = svm.SVC(kernel="linear", gamma="auto", probability=True)
    elif classifier == "rbf_svm":
        clf = svm.SVC(kernel="rbf", gamma="auto", probability=True)
    elif classifier == "sigmoid_svm":
        clf = svm.SVC(kernel="sigmoid", gamma="auto", probability=True)
    elif classifier == "decision_tree":
        clf = tree.DecisionTreeClassifier()
    elif classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=10)
    elif classifier == "GPC_RBF":
        kernel = 1.0 * RBF(length_scale=1.0)
        clf = GaussianProcessClassifier(kernel=kernel)
    elif classifier == "GPC_DOT":
        kernel = 1.0 * DotProduct(sigma_0=1.0)**2
        clf = GaussianProcessClassifier(kernel=kernel)
    elif classifier == "GPC_EXP":
        kernel = ExpSineSquared()
        clf = GaussianProcessClassifier(kernel=kernel)
    elif classifier == "GPC_MAT":
        kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
        clf = GaussianProcessClassifier(kernel=kernel)
        
    clf = clf.fit(train_set,train_labels)
    acc = clf.score(test_set,test_labels)
    y_pred = clf.predict_proba(test_set)
    y_score = clf.predict(test_set)
    
    if "svm" in classifier:
        pred = clf.decision_function(test_set)
        score_roc = roc_auc_score(test_labels, pred)
        fpr, tpr, thresholds = roc_curve(test_labels, pred)
    else:
        score_roc = roc_auc_score(test_labels, y_pred[:,1])
        fpr, tpr, _ = roc_curve(test_labels,y_pred[:,1])
    
    tn, fp, fn, tp = confusion_matrix(test_labels, y_score).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    
    base_fpr = np.linspace(0, 1, 101)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    return acc, score_roc, sn, sp, tpr, clf
        
def apply_feature_selection(df, labels, cutoff_pvalue=0.05, c_value=1):
    X=[]
    y=[]
    for key in list(df.index):
        X.append(df.loc[key])
        y.append(labels[key])
    X = np.array(X)
    y = np.hstack(y)
        
    remover = va(threshold=0.0)
    X = remover.fit_transform(X)
    variance_indices = remover.get_support(indices=True)
    
    f_scores, p_values = fc(X, y)
    critical_value = st.f.ppf(q=1-cutoff_pvalue, dfn=len(np.unique(y))-1, dfd=len(y)-len(np.unique(y)))
    
    best_indices=[]
    for index, p_value in enumerate(p_values):
        if f_scores[index]>=critical_value and p_value<cutoff_pvalue:
            best_indices.append(index)
    print("Best ANOVA features: "+str(len(best_indices)))
    
    df = df.iloc[:,variance_indices]
    best_columns = np.array(list(df.columns))[best_indices]
    best_features = np.array(list(df[best_columns].values))
    
    sel_ = SelectFromModel(LogisticRegression(C=c_value, penalty='l1', solver="liblinear"))
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

#load labels
labels=pd.read_excel(hypes["dataset_dir"]+"labels.xls")
            
pids = np.array(list(labels.index),dtype=str)
f_labels = np.array(list(labels.values),dtype=int)

iterations=100
splits={}  
for expindex in range(0,iterations):
    sss = StratifiedKFold(n_splits=4, shuffle=True)          
    kfolds=[]          
    for train_index, test_index in sss.split(pids, f_labels):
        kfolds.append([pids[train_index], pids[test_index]])
    splits["exp"+str(expindex)] = kfolds
pkl.dump(splits,open("chkp/splits.pkl","wb"))

results={}
r_results={}
d_results={}
t_results={}
c_results={}
rs_results={}
ds_results={}
rs_results={}
rts_results={}
ts_results={}
cs_results={}
mvs_results={}
failed=[]
all_splits=pkl.load(open("chkp/splits.pkl","rb"))
for classifier in ["GPC_RBF", "KNN", "decision_tree","poly_svm", "linear_svm", "rbf_svm", "sigmoid_svm"]:
    results_path = classifier
    os.mkdir("chkp/"+results_path)
    for model_name in list(deep.keys()):
        for experiment in ["ADJ-RESP"]: 
            ksplits={}
            for expindex in range(0,iterations):
                ksplits[experiment] = all_splits["exp"+str(expindex)]                             
                rg_kacc=[]
                rg_kauc=[]
                rg_ksn=[]
                rg_ksp=[]                
                g_kacc=[]
                g_kauc=[]
                g_ksn=[]
                g_ksp=[]                
                r_kacc=[]
                r_kauc=[]
                r_ksn=[]
                r_ksp=[]                
                d_kacc=[]
                d_kauc=[]
                d_ksn=[]
                d_ksp=[]                
                c_kacc=[]
                c_kauc=[]
                c_ksn=[]
                c_ksp=[]               
                ts_kacc=[]
                ts_kauc=[]
                ts_ksn=[]
                ts_ksp=[]               
                ds_kacc=[]
                ds_kauc=[]
                ds_ksn=[]
                ds_ksp=[]               
                rs_kacc=[]
                rs_kauc=[]
                rs_ksn=[]
                rs_ksp=[]               
                rts_kacc=[]
                rts_kauc=[]
                rts_ksn=[]
                rts_ksp=[]               
                cs_kacc=[]
                cs_kauc=[]
                cs_ksn=[]
                cs_ksp=[]               
                mvs_kacc=[]
                mvs_kauc=[]
                mvs_ksn=[]
                mvs_ksp=[]                   
                for index,split in enumerate(ksplits[experiment]):
                    tr_split=[]
                    tst_split=[]
                    for key in list(labels.index):
                        if key in list(split[0]):
                            tr_split.append(key)
                        elif key in list(split[1]):
                            tst_split.append(key)
                            
                    #standardize training set
                    radiomics_tr = robust_scale(radiomics.loc[tr_split],unit_variance=True,quantile_range=(10.0, 90.0))
                    radiomics_tr[radiomics_tr>1]=1
                    radiomics_tr[radiomics_tr<-1]=-1
                    radiomics_tr = pd.DataFrame(data=radiomics_tr,index=radiomics.loc[tr_split].index, columns=radiomics.loc[tr_split].columns)
                    transcriptomics_tr = robust_scale(transcriptomics.loc[tr_split],unit_variance=True,quantile_range=(10.0, 90.0))
                    transcriptomics_tr[transcriptomics_tr>1]=1
                    transcriptomics_tr[transcriptomics_tr<-1]=-1
                    transcriptomics_tr = pd.DataFrame(data=transcriptomics_tr,index=transcriptomics.loc[tr_split].index, columns=transcriptomics.loc[tr_split].columns)
                    deep_tr = robust_scale(deep[model_name].loc[tr_split],unit_variance=True,quantile_range=(10.0, 90.0))
                    deep_tr[deep_tr>1]=1
                    deep_tr[deep_tr<-1]=-1
                    
                    #standardize testing set
                    radiomics_tst = robust_scale(radiomics.loc[tst_split],unit_variance=True,quantile_range=(10.0, 90.0))
                    radiomics_tst[radiomics_tst>1]=1
                    radiomics_tst[radiomics_tst<-1]=-1
                    radiomics_tst = pd.DataFrame(data=radiomics_tst,index=radiomics.loc[tst_split].index, columns=radiomics.loc[tst_split].columns)
                    transcriptomics_tst = robust_scale(transcriptomics.loc[tst_split],unit_variance=True,quantile_range=(10.0, 90.0))
                    transcriptomics_tst[transcriptomics_tst>1]=1
                    transcriptomics_tst[transcriptomics_tst<-1]=-1
                    transcriptomics_tst = pd.DataFrame(data=transcriptomics_tst,index=transcriptomics.loc[tst_split].index, columns=transcriptomics.loc[tst_split].columns)
                    deep_tst = robust_scale(deep[model_name].loc[tst_split],unit_variance=True,quantile_range=(10.0, 90.0))
                    deep_tst[deep_tst>1]=1
                    deep_tst[deep_tst<-1]=-1
                    
                    columns=["f"+str(s) for s in deep[model_name].columns]
                    deep_tr = pd.DataFrame(data=deep_tr,index=deep[model_name].loc[tr_split].index, columns=columns)
                    deep_tst = pd.DataFrame(data=deep_tst,index=deep[model_name].loc[tst_split].index, columns=columns)
                    
                    #separate training labels from all labels                     
                    tr_labels={}
                    for pid in list(tr_split):
                        try:
                            tr_labels[pid] = labels[pid]
                        except:
                            continue
                    
                    try:
                        #feature selection
                        print("-feature selection, deep: "+model_name+" clf: "+classifier+" exp: "+experiment)                            
                        # try:
                        print("---Transcriptomics")
                        transcriptomics_feat, transcriptomics_coefs = apply_feature_selection(transcriptomics_tr, tr_labels, cutoff_pvalue=0.005, c_value=0.2)
                        print("---Radiomics")
                        radiomic_feat, radiomic_coefs = apply_feature_selection(radiomics_tr, tr_labels, cutoff_pvalue=0.05, c_value=0.9)
                        print("---Deep")
                        deep_feat, deep_coefs = apply_feature_selection(deep_tr, tr_labels, cutoff_pvalue=0.05, c_value=0.2)
                        print("---Clinical")
                        clinic_feat, clinic_coefs = apply_feature_selection(clinical.loc[tr_split], tr_labels, cutoff_pvalue=0.05, c_value=1.0)
                    except:
                        failed.append(experiment+"_"+model_name+"_"+"_"+classifier)
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
                    np.save(path+"/tr_split",np.array(tr_split,dtype=str))
                    np.save(path+"/tst_split",np.array(tst_split,dtype=str))                    
                        
                    selected_transcriptomics={}
                    selected_transcriptomics = get_features(selected_transcriptomics, transcriptomics_tr, transcriptomics_feat)
                    selected_transcriptomics = get_features(selected_transcriptomics, transcriptomics_tst, transcriptomics_feat)
                    
                    selected_radiomics={}
                    selected_radiomics = get_features(selected_radiomics, radiomics_tr, radiomic_feat)
                    selected_radiomics = get_features(selected_radiomics, radiomics_tst, radiomic_feat)
                    
                    selected_deep={}
                    selected_deep = get_features(selected_deep, deep_tr, deep_feat)
                    selected_deep = get_features(selected_deep, deep_tst, deep_feat)
                    
                    selected_clinical={}
                    selected_clinical = get_features(selected_clinical, clinical, clinic_feat)
                    
                    r_labels={}
                    r_patterns={}
                    d_patterns={}
                    c_patterns={}
                    t_patterns={}
                    for key in sorted(labels.keys()):
                        try:
                            t_patterns[key]=selected_transcriptomics[key]
                            r_patterns[key]=selected_radiomics[key]
                            d_patterns[key]=selected_deep[key]
                            c_patterns[key]=selected_clinical[key]
                        except:
                            print("Incomplete data for: "+key)
                            continue
                    
                    transcriptomics_score = get_composite_dict([t_patterns], [transcriptomics_coefs])
                    radiomics_score = get_composite_dict([r_patterns], [radiomic_coefs])
                    deep_score = get_composite_dict([d_patterns], [deep_coefs])
                    clinical_score = get_composite_dict([c_patterns], [clinic_coefs])
                    radiotranscriptomics_score = get_composite_dict([t_patterns, r_patterns, d_patterns], [transcriptomics_coefs, radiomic_coefs, deep_coefs])
                    multiview_scores = get_composite_dict([t_patterns, r_patterns, d_patterns, c_patterns], [transcriptomics_coefs, radiomic_coefs, deep_coefs, clinic_coefs])
                    
                    print("classification: "+classifier)
                    rg_acc, rg_auc, rg_sn, rg_sp, rg_tpr, rg_trained_clr = machine_learning({"clinical":c_patterns,"deep":d_patterns,"radiomics":r_patterns,"transcriptomics":t_patterns}, labels, {"clinical":[tr_split,tst_split],"deep":[tr_split,tst_split],"radiomics":[tr_split,tst_split],"transcriptomics":[tr_split,tst_split]}, classifier, multiview=True)
                    r_acc, r_auc, r_sn, r_sp, r_tpr, r_trained_clr = machine_learning(r_patterns, labels, [tr_split,tst_split], classifier, multiview=False)
                    d_acc, d_auc, d_sn, d_sp, d_tpr, d_trained_clr = machine_learning(d_patterns, labels, [tr_split,tst_split], classifier, multiview=False)
                    g_acc, g_auc, g_sn, g_sp, g_tpr, g_trained_clr = machine_learning(t_patterns, labels, [tr_split,tst_split], classifier, multiview=False)
                    c_acc, c_auc, c_sn, c_sp, c_tpr, c_trained_clr = machine_learning(c_patterns, labels, [tr_split,tst_split], classifier, multiview=False)
                    
                    ts_acc, ts_auc, ts_sn, ts_sp, ts_tpr, ts_trained_clr = machine_learning(transcriptomics_score, labels, [tr_split,tst_split], classifier, multiview=False)
                    rs_acc, rs_auc, rs_sn, rs_sp, rs_tpr, rs_trained_clr = machine_learning(radiomics_score, labels, [tr_split,tst_split], classifier, multiview=False)
                    ds_acc, ds_auc, ds_sn, ds_sp, ds_tpr, ds_trained_clr = machine_learning(deep_score, labels, [tr_split,tst_split], classifier, multiview=False)
                    cs_acc, cs_auc, cs_sn, cs_sp, cs_tpr, cs_trained_clr = machine_learning(clinical_score, labels, [tr_split,tst_split], classifier, multiview=False)
                    rts_acc, rts_auc, rts_sn, rts_sp, rts_tpr, rts_trained_clr = machine_learning(radiotranscriptomics_score, labels, [tr_split,tst_split], classifier, multiview=False)
                    mvs_acc, mvs_auc, mvs_sn, mvs_sp, mvs_tpr, mvs_trained_clr = machine_learning(multiview_scores, labels, [tr_split,tst_split], classifier, multiview=False)
                    
                    np.save(path+"/rg_tpr",np.array(rg_tpr))    
                    np.save(path+"/g_tpr",np.array(g_tpr))
                    np.save(path+"/r_tpr",np.array(r_tpr))
                    np.save(path+"/d_tpr",np.array(d_tpr))
                    np.save(path+"/c_tpr",np.array(c_tpr))
                    np.save(path+"/mvs_tpr",np.array(mvs_tpr))  
                    np.save(path+"/cs_tpr",np.array(cs_tpr)) 
                    np.save(path+"/rts_tpr",np.array(rts_tpr))
                    np.save(path+"/rs_tpr",np.array(rs_tpr))
                    np.save(path+"/ds_tpr",np.array(ds_tpr))
                    np.save(path+"/ts_tpr",np.array(ts_tpr)) 
                
                    np.save(path+"/rg_auc",np.array(rg_auc))    
                    np.save(path+"/g_auc",np.array(g_auc))
                    np.save(path+"/r_auc",np.array(r_auc))
                    np.save(path+"/d_auc",np.array(d_auc))
                    np.save(path+"/c_auc",np.array(c_auc))
                    np.save(path+"/mvs_auc",np.array(mvs_auc))  
                    np.save(path+"/cs_auc",np.array(cs_auc)) 
                    np.save(path+"/rts_auc",np.array(rts_auc))
                    np.save(path+"/rs_auc",np.array(rs_auc))
                    np.save(path+"/ds_auc",np.array(ds_auc))
                    np.save(path+"/ts_auc",np.array(ts_auc)) 
                    
                    pkl.dump(rg_trained_clr, open(path+"/rg_trained_clr.pkl","wb"))    
                    pkl.dump(r_trained_clr, open(path+"/r_trained_clr.pkl","wb"))
                    pkl.dump(d_trained_clr, open(path+"/d_trained_clr.pkl","wb"))
                    pkl.dump(g_trained_clr, open(path+"/g_trained_clr.pkl","wb"))
                    pkl.dump(c_trained_clr, open(path+"/c_trained_clr.pkl","wb"))  
                    pkl.dump(ts_trained_clr, open(path+"/ts_trained_clr.pkl","wb")) 
                    pkl.dump(rs_trained_clr, open(path+"/rs_trained_clr.pkl","wb"))
                    pkl.dump(ds_trained_clr, open(path+"/ds_trained_clr.pkl","wb"))
                    pkl.dump(cs_trained_clr, open(path+"/cs_trained_clr.pkl","wb"))
                    pkl.dump(rts_trained_clr, open(path+"/rts_trained_clr.pkl","wb"))                
                    pkl.dump(mvs_trained_clr, open(path+"/mvs_trained_clr.pkl","wb"))    
                    
                    ts_kacc.append(ts_acc)
                    ts_kauc.append(ts_auc)
                    ts_ksn.append(ts_sn)
                    ts_ksp.append(ts_sp)
                    
                    ds_kacc.append(ds_acc)
                    ds_kauc.append(ds_auc)
                    ds_ksn.append(ds_sn)
                    ds_ksp.append(ds_sp)
                    
                    rs_kacc.append(rs_acc)
                    rs_kauc.append(rs_auc)
                    rs_ksn.append(rs_sn)
                    rs_ksp.append(rs_sp)
                    
                    rts_kacc.append(rts_acc)
                    rts_kauc.append(rts_auc)
                    rts_ksn.append(rts_sn)
                    rts_ksp.append(rts_sp)
                    
                    cs_kacc.append(cs_acc)
                    cs_kauc.append(cs_auc)
                    cs_ksn.append(cs_sn)
                    cs_ksp.append(cs_sp)
                    
                    mvs_kacc.append(mvs_acc)
                    mvs_kauc.append(mvs_auc)
                    mvs_ksn.append(mvs_sn)
                    mvs_ksp.append(mvs_sp)
                    
                    rg_kacc.append(rg_acc)
                    rg_kauc.append(rg_auc)
                    rg_ksn.append(rg_sn)
                    rg_ksp.append(rg_sp)
                    
                    g_kacc.append(g_acc)
                    g_kauc.append(g_auc)
                    g_ksn.append(g_sn)
                    g_ksp.append(g_sp)
                    
                    r_kacc.append(r_acc)
                    r_kauc.append(r_auc)
                    r_ksn.append(r_sn)
                    r_ksp.append(r_sp)
                    
                    d_kacc.append(d_acc)
                    d_kauc.append(d_auc)
                    d_ksn.append(d_sn)
                    d_ksp.append(d_sp)
                                    
                    c_kacc.append(c_acc)
                    c_kauc.append(c_auc)
                    c_ksn.append(c_sn)
                    c_ksp.append(c_sp)
                    
                    
                #scores metrics
                ts_kacc=np.array(ts_kacc)
                ts_kauc=np.array(ts_kauc)
                ts_ksn=np.array(ts_ksn)
                ts_ksp=np.array(ts_ksp)
                
                ds_kacc=np.array(ds_kacc)
                ds_kauc=np.array(ds_kauc)
                ds_ksn=np.array(ds_ksn)
                ds_ksp=np.array(ds_ksp)
                
                rs_kacc=np.array(rs_kacc)
                rs_kauc=np.array(rs_kauc)
                rs_ksn=np.array(rs_ksn)
                rs_ksp=np.array(rs_ksp)
                
                rts_kacc=np.array(rts_kacc)
                rts_kauc=np.array(rts_kauc)
                rts_ksn=np.array(rts_ksn)
                rts_ksp=np.array(rts_ksp)
                
                cs_kacc=np.array(cs_kacc)
                cs_kauc=np.array(cs_kauc)
                cs_ksn=np.array(cs_ksn)
                cs_ksp=np.array(cs_ksp)
                
                mvs_kacc=np.array(mvs_kacc)
                mvs_kauc=np.array(mvs_kauc)
                mvs_ksn=np.array(mvs_ksn)
                mvs_ksp=np.array(mvs_ksp)
                    
                #classification metrics
                rg_kacc=np.array(rg_kacc)
                rg_kauc=np.array(rg_kauc)
                rg_ksn=np.array(rg_ksn)
                rg_ksp=np.array(rg_ksp)
                
                g_kacc=np.array(g_kacc)
                g_kauc=np.array(g_kauc)
                g_ksn=np.array(g_ksn)
                g_ksp=np.array(g_ksp)
                
                r_kacc=np.array(r_kacc)
                r_kauc=np.array(r_kauc)
                r_ksn=np.array(r_ksn)
                r_ksp=np.array(r_ksp)
                
                d_kacc=np.array(d_kacc)
                d_kauc=np.array(d_kauc)
                d_ksn=np.array(d_ksn)
                d_ksp=np.array(d_ksp)
                
                c_kacc=np.array(c_kacc)
                c_kauc=np.array(c_kauc)
                c_ksn=np.array(c_ksn)
                c_ksp=np.array(c_ksp)                
                
                # csv metrics
                results["multi-view "+classifier+" "+experiment+" "+model_name+str(expindex)] = pd.Series({"ACC_MEAN":rg_kacc.mean(),"ACC_STD":rg_kacc.std(),"AUC_MEAN":rg_kauc.mean(),"AUC_STD":rg_kauc.std(),"SN_MEAN":rg_ksn.mean(),"SN_STD":rg_ksn.std(),"SP_MEAN":rg_ksp.mean(),"SP_STD":rg_ksp.std()})
                r_results["radiomics "+classifier+" "+experiment+" "+model_name+str(expindex)] = pd.Series({"ACC_MEAN":r_kacc.mean(),"ACC_STD":r_kacc.std(),"AUC_MEAN":r_kauc.mean(),"AUC_STD":r_kauc.std(),"SN_MEAN":r_ksn.mean(),"SN_STD":r_ksn.std(),"SP_MEAN":r_ksp.mean(),"SP_STD":r_ksp.std()})
                d_results["deep "+classifier+" "+experiment+" "+model_name+str(expindex)] = pd.Series({"ACC_MEAN":d_kacc.mean(),"ACC_STD":d_kacc.std(),"AUC_MEAN":d_kauc.mean(),"AUC_STD":d_kauc.std(),"SN_MEAN":d_ksn.mean(),"SN_STD":d_ksn.std(),"SP_MEAN":d_ksp.mean(),"SP_STD":d_ksp.std()})
                t_results["transcriptomics "+classifier+" "+experiment+" "+model_name+str(expindex)] = pd.Series({"ACC_MEAN":g_kacc.mean(),"ACC_STD":g_kacc.std(),"AUC_MEAN":g_kauc.mean(),"AUC_STD":g_kauc.std(),"SN_MEAN":g_ksn.mean(),"SN_STD":g_ksn.std(),"SP_MEAN":g_ksp.mean(),"SP_STD":g_ksp.std()})
                c_results["clinical "+classifier+" "+experiment+" "+model_name+str(expindex)] = pd.Series({"ACC_MEAN":c_kacc.mean(),"ACC_STD":c_kacc.std(),"AUC_MEAN":c_kauc.mean(),"AUC_STD":c_kauc.std(),"SN_MEAN":c_ksn.mean(),"SN_STD":c_ksn.std(),"SP_MEAN":c_ksp.mean(),"SP_STD":c_ksp.std()})
                
                ds_results["deep score "+classifier+" "+experiment+" "+model_name+str(expindex)]= pd.Series({"ACC_MEAN":ds_kacc.mean(),"ACC_STD":ds_kacc.std(),"AUC_MEAN":ds_kauc.mean(),"AUC_STD":ds_kauc.std(),"SN_MEAN":ds_ksn.mean(),"SN_STD":ds_ksn.std(),"SP_MEAN":ds_ksp.mean(),"SP_STD":ds_ksp.std()})
                rs_results["radiomic score "+classifier+" "+experiment+" "+model_name+str(expindex)]= pd.Series({"ACC_MEAN":rs_kacc.mean(),"ACC_STD":rs_kacc.std(),"AUC_MEAN":rs_kauc.mean(),"AUC_STD":rs_kauc.std(),"SN_MEAN":rs_ksn.mean(),"SN_STD":rs_ksn.std(),"SP_MEAN":rs_ksp.mean(),"SP_STD":rs_ksp.std()})
                rts_results["radiotranscriptomic score "+classifier+" "+experiment+" "+model_name+str(expindex)]= pd.Series({"ACC_MEAN":rts_kacc.mean(),"ACC_STD":rts_kacc.std(),"AUC_MEAN":rts_kauc.mean(),"AUC_STD":rts_kauc.std(),"SN_MEAN":rts_ksn.mean(),"SN_STD":rts_ksn.std(),"SP_MEAN":rts_ksp.mean(),"SP_STD":rts_ksp.std()})
                ts_results["transcriptomics score "+classifier+" "+experiment+" "+model_name+str(expindex)]= pd.Series({"ACC_MEAN":ts_kacc.mean(),"ACC_STD":ts_kacc.std(),"AUC_MEAN":ts_kauc.mean(),"AUC_STD":ts_kauc.std(),"SN_MEAN":ts_ksn.mean(),"SN_STD":ts_ksn.std(),"SP_MEAN":ts_ksp.mean(),"SP_STD":ts_ksp.std()})
                cs_results["clinical score "+classifier+" "+experiment+" "+model_name+str(expindex)]= pd.Series({"ACC_MEAN":cs_kacc.mean(),"ACC_STD":cs_kacc.std(),"AUC_MEAN":cs_kauc.mean(),"AUC_STD":cs_kauc.std(),"SN_MEAN":cs_ksn.mean(),"SN_STD":cs_ksn.std(),"SP_MEAN":cs_ksp.mean(),"SP_STD":cs_ksp.std()})
                mvs_results["multi-view scores "+classifier+" "+experiment+" "+model_name+str(expindex)]= pd.Series({"ACC_MEAN":mvs_kacc.mean(),"ACC_STD":mvs_kacc.std(),"AUC_MEAN":mvs_kauc.mean(),"AUC_STD":mvs_kauc.std(),"SN_MEAN":mvs_ksn.mean(),"SN_STD":mvs_ksn.std(),"SP_MEAN":mvs_ksp.mean(),"SP_STD":mvs_ksp.std()})
                       
multiview_results = pd.DataFrame.from_dict(results, orient="index")
r_results = pd.DataFrame.from_dict(r_results, orient="index")
d_results = pd.DataFrame.from_dict(d_results, orient="index")
t_results = pd.DataFrame.from_dict(t_results, orient="index")
c_results = pd.DataFrame.from_dict(c_results, orient="index")

ds_results = pd.DataFrame.from_dict(ds_results, orient="index")
rs_results = pd.DataFrame.from_dict(rs_results, orient="index")
rts_results = pd.DataFrame.from_dict(rts_results, orient="index")
ts_results = pd.DataFrame.from_dict(ts_results, orient="index")
cs_results = pd.DataFrame.from_dict(cs_results, orient="index")
mvs_results = pd.DataFrame.from_dict(mvs_results, orient="index")

multiview_results.to_csv("chkp/"+results_path+"/multi-view_results.csv")
r_results.to_csv("chkp/"+results_path+"/radiomics_results.csv")
d_results.to_csv("chkp/"+results_path+"/deep_results.csv")
t_results.to_csv("chkp/"+results_path+"/transcriptomics_results.csv")
c_results.to_csv("chkp/"+results_path+"/clinical_results.csv")

ds_results.to_csv("chkp/"+results_path+"/deep_scores_results.csv")
rs_results.to_csv("chkp/"+results_path+"/radiomic_scores_results.csv")
rts_results.to_csv("chkp/"+results_path+"/radiotranscriptomic_scores_results.csv")
ts_results.to_csv("chkp/"+results_path+"/transcriptomics_scores_results.csv")
cs_results.to_csv("chkp/"+results_path+"/clinical_scores_results.csv")
mvs_results.to_csv("chkp/"+results_path+"/multiview_scores_results.csv")