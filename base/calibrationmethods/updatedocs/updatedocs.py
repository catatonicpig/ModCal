# -*- coding: utf-8 -*-
"""
The purpose of this file is to automatically add documentation and guidence into our methods 
folders. There was no existing functionality that this author could find.
"""
import copy
import os
import shutil
import time
import re

methodfiles = os.listdir("../")

for k in range(len(methodfiles)-1, -1, -1):
   if not methodfiles[k].endswith('.py'):
        print('skipping non-.py file ' + methodfiles[k])
        methodfiles.remove(methodfiles[k])


docfile = open("docinfo.py", "r")
doclines = docfile.readlines()
nameinsideitself = False
docstringname = []
docstringstarts = []
docstringends = []
i = 0
while i < len(doclines):
    count = (len(doclines[i]) - len(doclines[i].replace("\"\"\"","")))
    startdocstring = -1
    if count % 6 == 3:
        startdocstring = copy.copy(i)
    elif count % 6 != 0:
        replacefile = False
        print('there seems to be an error somewhere interpreting this file.')
    enddocstring = -1
    if startdocstring >= 0:
        if doclines[startdocstring-1].startswith('[') and doclines[startdocstring-1].endswith(']\n'):
            kve = 1+1
        else:
            startdocstring = -1
    if startdocstring >= 0:
        for j in range(i+1, min(i+2000,len(doclines))):
            count = (len(doclines[j]) - len(doclines[j].replace("\"\"\"","")))
            if count % 6 == 3:
                enddocstring = copy.copy(j)
                break
            elif count % 6 != 0:
                replacefile = False
                print('there seems to be an error somewhere interpreting this file.')
        docstringstarts.append(1*startdocstring)
        docstringends.append(1*enddocstring)
        docstringname += [doclines[startdocstring-1][1:-2]]
        i = 1+copy.copy(enddocstring)
    else:
        i = i+1

for l in range(len(methodfiles)-1, -1, -1):
    formatfile = open("../" + methodfiles[l], "r")
    newfile = open(methodfiles[l], "w")
    searchlines = formatfile.readlines()
    i = 0
    inquote = False
    replacefile = True
    didonce = False
    while i < len(searchlines):
        filestar = None
        count = (len(searchlines[i]) - len(searchlines[i].replace("\"\"\"","")))
        if count % 6 == 3:
            startdocstring = copy.copy(i)
        elif count % 6 != 0:
            replacefile = False
            print('there seems to be an error somewhere interpreting this file. \n I will not replace it.')
        else:
            startdocstring = -1
            newfile.write(searchlines[i])
            i = i + 1
        enddocstring = -1
        if startdocstring >= 0:
            for j in range(i+1, min(i+1000,len(searchlines))):
                count = (len(searchlines[j]) - len(searchlines[j].replace("\"\"\"","")))
                if count % 6 == 3:
                    enddocstring = copy.copy(j)
                    break
                elif count % 6 != 0:
                    replacefile = False
                    print('there seems to be an error somewhere interpreting this file.  \n I will not replace it.')
        kstar = None
        if enddocstring > startdocstring:
            for k in range(0,len(docstringname)):
                for j in range(startdocstring,enddocstring):
                    if (docstringname[k] in searchlines[j]):
                        kstar = copy.copy(k)
                        break
                if kstar is not None:
                    break
            if kstar is not None:
                didonce = True
                for j in range(docstringstarts[kstar], 1+docstringends[kstar]):
                    newfile.write(doclines[j])
                docfile.close()
            else:
                for j in range(startdocstring,1+enddocstring):
                    newfile.write(searchlines[j])
            i = enddocstring + 1
        elif startdocstring > 0:
            replacefile = False
            print('there seems to be an error somewhere interpreting this file.  \n I will not replace it.')
            break
    newfile.close()
    formatfile.close()
    if replacefile and didonce:
        timestr = re.sub('\D', '', str(time.gmtime()))
        tempname = timestr + methodfiles[l] + 'temp'
        os.rename('../' + methodfiles[l], 'temp/' + tempname)
        try:
            print('updated docstrings in ' + methodfiles[l])
            os.rename(methodfiles[l], '../' + methodfiles[l])
        except:
            print('failed to move ' + methodfiles[l])
            os.rename('temp/' + tempname, methodfiles[l])
    else:
        timestr = re.sub('\D', '', str(time.gmtime()))
        tempname = timestr + methodfiles[l] + 'temp'
        os.rename(methodfiles[l], 'temp/' + tempname)
        print('could not find docstrings to replace in ' + methodfiles[l])
