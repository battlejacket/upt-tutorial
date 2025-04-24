def setDict(parameters, parameterDict):
    for i, key in enumerate(parameterDict.keys()):
        parameterDict[key] = parameterDict[key](parameters[i])
    return parameterDict

def readParametersFromFileName(fileName, parameterDict, divider='_', generateNameString=False):
    parameters = fileName.replace('.csv','').replace(',','.').split('_')
    parameterDict = setDict(parameters, parameterDict)
    return parameterDict