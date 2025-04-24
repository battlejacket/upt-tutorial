def setDict(parameters, parameterDict):
    for i, key in enumerate(parameterDict.keys()):
        parameterDict[key] = parameterDict[key](parameters[i])
    return parameterDict

def readParametersFromFileName(fileName, parameterDef, divider='_', generateNameString=False):
    parameterDict = parameterDef.copy()
    parameters = fileName.replace('.csv','').replace(',','.').split('_')
    parameterDict = setDict(parameters, parameterDict)
    return parameterDict