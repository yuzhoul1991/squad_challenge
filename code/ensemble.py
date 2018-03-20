import os
import sys
import json
import io
from collections import Counter

num_of_models = sys.argv[1]
print("number of models: ", num_of_models)

predictions = []

for i in range(int(num_of_models)):
    model_pred = []
    pred_file = "prediction{0}.json".format(i)
    assert(os.path.exists(pred_file))
    pred = json.load(open(pred_file, 'r'))
    for k, v in pred.iteritems():
        temp = [k, v]
        model_pred.append(temp)
    predictions.append(model_pred)

uuid2ans = {}
for preds in zip(*predictions):
    uuids = map(lambda x: x[0], preds)
    ans = map(lambda x: x[1], preds)
    assert(len(set(uuids)) == 1)
    counts = Counter(ans)
    most_common = counts.most_common()[0][0]
    uuid2ans[uuids[0]] = most_common

with io.open('predictions.json', 'w', encoding='utf-8') as fh:
    fh.write(unicode(json.dumps(uuid2ans, ensure_ascii=False)))
    print "Writing do predictions.json done"
