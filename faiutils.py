"""
Various convenience functions for Fastai.
"""
import csv
import datetime
import difflib
import gzip
import itertools
import json
import numbers
import numpy as np
import os
import pickle
import random
import sys

try :
    import torch
    torch_avail = True
except :
    print('PyTorch not available - randomSeedForTraining will not work!')
    torch_avail = False


def randomSeedForTraining(seed) :
    """
    Initialize all random number generators used in fastai training with a given seed.
    Use when you need repeatable training.
    """
    if not torch_avail :
        raise Exception('PyTorch not available!')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.cuda.manual_seed(seed_value)
    # torch.backends.cudnn.benchmark = False


# Convenience file functions.
def openGzipOrText(fPath,encoding=None) :
    """Opens a file for reading as text, uncompressing it on read if the path ends in .gz"""
    if str(fPath).lower().endswith('.gz') :
        return gzip.open(fPath,'rt',encoding=encoding)
    else :
        return open(fPath,'rt',encoding=encoding)
def needToRefresh(dependentFPaths, fromFPath) :
    """
    Says if a file or list of files that depend on another needs to be refreshed.
    Returns True if any of the dependent files don't exist or are older.
    """
    if not isinstance(dependentFPaths,list) :
        dependentFPaths = [dependentFPaths]
    return any (((not os.path.exists(dependentFPath))
                    or os.path.getmtime(fromFPath) >= os.path.getmtime(dependentFPath))
                for dependentFPath in dependentFPaths)
def saveToPkl(fPath,ob) :
    """Save an object in pickled format to a file."""
    with open(fPath,'wb') as f :
        pickle.dump(ob,f,2)
def loadFromPkl(fPath) :
    """Load and return a pickled object from a file."""
    with open(fPath,'rb') as f :
        return pickle.load(f)


def exportFromNotebook(notebookPath, outFNames, outFPath=None, encoding='utf-8') :
    """
    Routine to auto-generate a python script from Jupyter notebook cells.
    Exports code cells that start with
        # export name1 name2 ...
    where one of the names is in the supplied list of outFNames.
    Use outFNames='export' to export all the code cells marked for export.
    Saves to outFPath if given, else returns the output text as a string.
    """
    if isinstance(outFNames,str) :
        outFNames = [outFNames]
    outLnList = ['# auto-generated from '+str(notebookPath)+' at '
                 +datetime.datetime.now().isoformat().replace('T',' ')+'\n',
                 'running_exported_scripts = {'
                    + ','.join(repr(outFName) for outFName in outFNames)
                    + '}\n']
    outFNames = set(outFNames)
    for sourceLnList in getCodeCells(notebookPath) :
        nonEmptyLns = [ln for ln in sourceLnList if ln.strip()!='']
        if len(nonEmptyLns) == 0 :
            continue
        if not nonEmptyLns[-1].endswith('\n') :
            nonEmptyLns[-1] += '\n'
        firstLn = nonEmptyLns[0].split()
        if firstLn[:1] == ['#export'] :
            firstLn[:1] = ['#','export']
        if (firstLn[:2] == ['#','export']
                and len(outFNames.intersection(firstLn)) > 0) :
            outLnList.extend(nonEmptyLns)
    outText = ''.join(outLnList)
    if outFPath is not None :
        with open(outFPath,'wt',encoding=encoding) as f :
            f.write(outText)
    else :
        return outText
def getCodeCells(nbPath, encoding='utf-8') :
    with open(nbPath,'rt',encoding=encoding) as f :
        j = json.load(f)
        return [c['source'] for c in j['cells'] if c['cell_type']=='code']
def diffNBCells(nb1Path, nb2Path, cell1Inds, cell2Inds=None, encoding='utf-8') :
    if isinstance(cell1Inds,int) :
        cell1Inds = [cell1Inds]
    if cell2Inds is None :
        cell2Inds = cell1Inds
    elif isinstance(cell2Inds,int) :
        cell2Inds = [cell2Inds]
    nb1CodeCells = getCodeCells(nb1Path)
    nb2CodeCells = getCodeCells(nb2Path)
    print('comparing',nb1Path,'cells',cell1Inds,'with',nb2Path,cell2Inds)
    for c1Ind,c2Ind in zip(cell1Inds,cell2Inds) :
        diffLines = list(difflib.context_diff(nb1CodeCells[c1Ind], nb2CodeCells[c2Ind]))
        print('*** [{}] {}= [{}]'.format(c1Ind, '=' if len(diffLines)==0 else '!', c2Ind))
        sys.stdout.writelines(diffLines[3:])


# Code to automate naming models based on training hyperparameters,
# and keeping multiple versions from different stages of training.
modelDirPath = None # This should be set to the directory path used
                    # for model files before using these functions!
def getModelFNameFromHyperPars(pref, *hyperParsList, suff=None) :
    """
    Generate a model filename from a list [hyperpar_entry ... ]
    where each hyperpar_entry is either a dict of hyperparameter values
    or a tuple (hyperpars_dict, default_hyperpars_dict).
    In the second case, only the values different from the default
    are used to generate the filename.

    The filename is in the format pref_key1_val1_key2_val2...
    """
    res = pref
    for hyperPars in hyperParsList :
        if isinstance(hyperPars,tuple) or isinstance(hyperPars,list) :
            hyperPars, defHyperPars = hyperPars
        else :
            defHyperPars = {}
        for k in sorted(hyperPars.keys()) :
            if hyperPars[k] != defHyperPars.get(k) :
                res += '_{}_{}'.format(k,hyperPars[k]).replace('.','_')
    return res+(('_'+suff) if suff else '')
def checkModelDirPath() :
    if modelDirPath is None :
        raise Exception('faiutils.modelDirPath not initialized to model directory path!')
def modelExists(modelFName) :
    "Says if the first version of the model exists yet."
    checkModelDirPath()
    return (modelDirPath/(modelFName+'.pth')).exists()
def getModelVersions(modelFName) :
    """
    Returns a list of the existing model versions containing modelFName,
    plus the first version that doesn't yet exist.
    These should be in the format:
        [modelFName+'.pth', modelFName+'_1.pth', modelFName+'_2.pth', ... ]
    """
    checkModelDirPath()
    i = 0
    res = [modelFName]
    while (modelDirPath/(res[-1]+'.pth')).exists() :
        i += 1
        res.append(modelFName + '_' + str(i))
    return res
def loadModelVersion(learn,modelFName,i,printLoadingMessage=True) :
    """
    Loads the i'th model version containing modelFName.
    Allows using negative values for i to index back from the latest version.
    """
    modelVersions = getModelVersions(modelFName)[:-1]
    if i >= 0 :
        versionAvailable = i < len(modelVersions)
    else :
        versionAvailable = -i <= len(modelVersions)
    if versionAvailable :
        if printLoadingMessage :
            print('Loading',modelVersions[i])
        learn.load(modelVersions[i])
    elif printLoadingMessage :
        print('Version not available to load!')
def loadLatestModelVersion(learn,modelFName,printLoadingMessage=True) :
    "Loads the latest model version containing modelFName."
    loadModelVersion(learn,modelFName,-1,printLoadingMessage)
def removeModelVersions(modelFName, backToVer) :
    """
    Removes all model versions containing modelFName, up to and including backToVer
    (0 - remove all versions; -3: remove last 3 versions).
    """
    modelVersions = getModelVersions(modelFName)[:-1]
    versionsToRemove = modelVersions[backToVer:]
    if len(versionsToRemove) == 0 :
        print('No versions >=',backToVer,'available to remove!')
    else :
        for modelName in reversed(versionsToRemove) :
            print('Removing',modelName)
            (modelDirPath/(modelName+'.pth')).unlink()
def removeLatestModelVersion(modelFName) :
    "Removes the latest model version containing modelFName."
    removeModelVersions(modelFName,-1)
def removeAllModelVersions(modelFName) :
    "Removes all model versions containing modelFName."
    removeModelVersions(modelFName,0)
def saveNextModelVersion(learn,modelFName) :
    "Saves model to the next version containing modelFName."
    modelVersions = getModelVersions(modelFName)
    print('Saving',modelVersions[-1])
    learn.save(modelVersions[-1])
def overwriteModelVersion(learn,modelFName) :
    "Saves model to the last version containing modelFName, overwriting it."
    modelVersions = getModelVersions(modelFName)
    if len(modelVersions) == 1 :
        print('Saving',modelVersions[-1])
    else :
        del modelVersions[-1]
        print('Overwriting',modelVersions[-1])
    learn.save(modelVersions[-1])


# Code to do basic hyperparameter search (random or systematic).
def doHPSearch(parSel, trainFunc, resultsFName='hpSearch.pkl', nTrials=None, randomize=True) :
    """
    Does a hyperparameter search (random or systematic), saving the results.
    The parSel argument should be a dict; for each k,v in the dict, a hyperparameter
    value for k will be selected based on v. If v is a list starting with 'select',
    one of the members is chosen. Otherwise, v is passed through unchanged. Example:
        parSel = {
            'lr': ['select',0.003,0.002],
            'nEpochs': ['select',20,30],
            'max_lighting': 0.1,
        }
    k can also be a tuple of keys, in which case all the members of v should be
    tuples of the same length. Example:
        parSel = {
            ('lr','nEpochs'): ['select', (0.003,20), (0.002,30)],
            'max_lighting': 0.1,
        }

    The trainFunc argument should be a function that takes a hyperparameter
    selection supplied as keyword arguments, does the specified training,
    and returns a dict giving the results of interest - in particular,
    it should return 'final': <final main metric value>.

    A hyperparameter search can be interrupted and then called again -
    it will then continue searching selections not already tried.

    If nTrials is given it sets a limit on the number of selections to try.
    """
    checkModelDirPath()
    resultsFPath = modelDirPath/resultsFName
    if resultsFPath.exists() :
        print('appending to',resultsFPath)
        resList = loadFromPkl(resultsFPath)
    else :
        print('creating',resultsFPath)
        resList = []
    alreadyTried = dict((hashifyDict(hPars),result) for hPars,result in resList)
    i = 0
    for chosenPars in iterHPars(parSel,randomize) :
        if nTrials is not None and i>=nTrials :
            print('Limit of',nTrials,'trials exceeded.')
            return
        hChosenPars = hashifyDict(chosenPars)
        if hChosenPars in alreadyTried and alreadyTried[hChosenPars].get('status')=='OK' :
            print('already tried',chosenPars)
            continue
        i += 1
        print('trying selection',i,'-',chosenPars)
        try :
            result = dict(trainFunc(**chosenPars), status='OK')
        except Exception as e :
            result = {'status': 'Unable to train - '+str(e)}
        print('result',result)
        alreadyTried[hChosenPars] = result
        resList.append((chosenPars,result))
        saveToPkl(resultsFPath,resList)
    print('seached all possible selections!')
def hashifyValue(v) :
    """
    Try to convert v into a hashable type. This is used on hyperparameter
    selections so we can check whether we've already tried a selection.
    """
    if isinstance(v,list) or isinstance(v,tuple) :
        return tuple(hashifyValue(vv) for vv in v)
    if isinstance(v,set) :
        return frozenset(hashifyValue(vv) for vv in v)
    if isinstance(v,dict) :
        return hashifyDict(v)
    if isinstance(v,numbers.Number) :
        return v
    return str(v)
def hashifyDict(m) :
    return frozenset((k,hashifyValue(v)) for k,v in m.items())
def showResults(resList='hpSearch.pkl', nToShow=None,
                keyFunc = lambda res : res[1].get('final',1e6),
                printResults=True, shortFormat=True, csvOutFPath=None, 
                **kwargs) :
    """
    Shows results in a hyperparameter search saved by doRandSearch.

    Only results that pass the filter specified by kwargs are shown - for each
    k,v in kwargs, includes only result pairs for which hyperpars[k] == v, or,
    if v is a list starting with 'select', hyperpars[k] in v.

    Sorts results by keyFunc(resultPair), where resultPair is (hyperpars,results)
    as saved by doRandSearch.

    Optionally writes results to csv if csvOutFPath is not None.
    """
    checkModelDirPath()
    returnResList = False
    if isinstance(resList,str) :
        resultsFPath = modelDirPath/resList
        if not resultsFPath.exists() :
            raise Exception(str(resultsFPath)+' not found!')
        resList = loadFromPkl(resultsFPath)
        returnResList = True
    resList = filterResults(resList,kwargs)
    if keyFunc is not None :
        resList.sort(key=keyFunc)
    resList = resList if nToShow is None else resList[:nToShow]
    if len(resList)==0 :
        if printResults :
            print('No results to show!')
        return []
    firstRes = resList[0]
    if printResults :
        singleValueKeys = set(firstRes[0].keys())
        for res in resList[1:] :
            singleValueKeys.intersection_update(res[0].keys())
        firstKeyValue = dict((k,hashifyValue(firstRes[0][k])) for k in singleValueKeys)
        for res in resList[1:] :
            for k,hv in firstKeyValue.items() :
                if hv != hashifyValue(res[0][k]) :
                    singleValueKeys.discard(k)
        if len(singleValueKeys) > 0 :
            commonHPars = sorted(item for item in firstRes[0].items()
                                    if item[0] in singleValueKeys)
            if shortFormat :
                print('Common:',
                        ', '.join('{}: {}'.format(k,v) for k,v in commonHPars))
            else :
                print('Common hyperparameters:',commonHPars)
        for i,res in enumerate(resList) :
            itemsToShow = sorted(item for item in res[0].items()
                                    if item[0] not in singleValueKeys)
            if shortFormat :
                print('{:>3}:'.format(i),
                        ', '.join('{}: {}'.format(k,v) for k,v in itemsToShow),
                        '->', res[1].get('final',res[1].get('status','ERR')))
            else :
                print('Trial {}:'.format(i),itemsToShow)
                print('Result: ',res[1])
    if csvOutFPath is not None :
        allKeys = set(firstRes[0].keys())
        for res in resList[1:] :
            allKeys.update(res[0].keys())
        allKeys = sorted(allKeys)
        with open(csvOutFPath,'w',newline='') as f :
            csvw = csv.writer(f)
            csvw.writerow(allKeys+['result'])
            for res in resList :
                csvw.writerow([str(res[0].get(k,'')) for k in allKeys]
                                + [keyFunc(res) if keyFunc else str(res[1])])
    if returnResList :
        return resList
def iterHPars(parSel, randomize=True) :
    """
    Iterate through all possible selections of hyperparameters specified by parSel,
    either in systematic or in random order.

    parSel is a dict specifying possible hyperparameter selections as described in doHPSearch.
    """
    sortedParSelKeys = sorted(parSel.keys(), key = lambda k : tuple(k))
        # each key in parSel can be a single hyperparameter key or a tuple of keys
        # tuple(k) converts single keys to 1-length tuples so we can sort everything together
    parPossibleVals = [singleHParList(k,parSel[k],randomize)
                        for k in sortedParSelKeys]
    for parList in itertools.product(*parPossibleVals) :
        yield makeParsDict(parList)
sysRng = random.SystemRandom()
def singleHParList(k,v,randomize) :
    if isinstance(v,list) and v[:1]==['select'] :
        lis = [(k,vv) for vv in v[1:]]
        if randomize :
            sysRng.shuffle(lis)
        return lis
    else :
        return [(k,v)]
def makeParsDict(pairList) :
    res = {}
    for k,v in pairList :
        if isinstance(k,tuple) :  # k is (k1, k2, ... ), v is (v1, v2, ... )
            res.update(zip(k,v))
        else :
            res[k] = v
    return res
def filterResults(resList,kwargs) :
    """
    Filter a list of result pairs (hyperpars,result) based on kwargs. For each
    k,v in kwargs, includes only result pairs for which hyperpars[k] == v, or,
    if v is a list starting with 'select', hyperpars[k] in v.
    """
    return [res for res in resList
            if all(includeResult(res,k,v) for k,v in kwargs.items())]
def includeResult(resPair,k,v) :
    hpars,_ = resPair
    if k in hpars :
        if isinstance(v,list) and v[:1]==['select'] :
            return hpars[k] in v[1:]
        else :
            return hpars[k] == v
    return False
