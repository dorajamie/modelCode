from operator import itemgetter

ids = [2616,1]
parentDict = {
    1:123,
    2:444,
    2616:333
}

correspondingParentsIds = list(itemgetter(*ids)(parentDict))
print(correspondingParentsIds)