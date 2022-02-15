import json

maxResponses = 0
maxIPs = 0

for curFileNum in range(0, 2):

    print("Current file number out of 190: " +str(curFileNum))
    curJSONFile = r'C:\Users\jacob\OneDrive\Documents\Princeton\Masters\Research\DataProcessingWork\SmallSample\JSON_Part_' + str(curFileNum) +'.txt'

    try:
        with open(curJSONFile, "r") as f:
            data = json.loads(f.read())

            for dictJSON in data:

                responseList = dictJSON["response"]

                curResponses = len(responseList)

                if curResponses > maxResponses:
                    maxResponses = curResponses
                    print("maxResponses: " + str(maxResponses))

                for response in responseList:

                    if response['has_type_a'] == True:

                        curIPs = len(response['response'])

                        if curIPs > maxIPs:
                            maxIPs = curIPs
                            print("maxIPs: " +str(curIPs))

    except:
        raise Exception("Reading " +curJSONFile+ " encountered an error")

    print("maxResponsesEnd: " +str(maxResponses))
    print("maxIPsEnd: " + str(maxIPs))