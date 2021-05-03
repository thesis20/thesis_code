import pandas as pd

dataset_path = 'Data/Frappe/'

user_id_column = 'user'
item_id_column = 'item'
sideinfo_columns =  ['cost']

out = pd.read_csv(dataset_path + 'out.txt')
train = pd.read_csv(dataset_path + 'train.txt')
test = pd.read_csv(dataset_path + 'test.txt')



user_list = ''
item_list = ''
user_dict = {}
item_dict = {}
for index, userId in enumerate(out[user_id_column].unique()):
    user_list += str(userId) + ' ' + str(index) + '\n'
    user_dict[userId] = index
for index, itemId in enumerate(out[item_id_column].unique()):
    item_list += 'item' + str(itemId) + ' ' + \
        str(index) + ' ' + str(index) + '\n'
    item_dict[itemId] = index

f = open("user_list.txt", "w")
f.write(user_list)
f.close()
f = open("item_list.txt", "w")
f.write(item_list)
f.close()

counter = 0
entity_list = ''
entity_dict = {}
for itemId in out[item_id_column].unique():
    if itemId == 'NULL':
        continue
    entity_list += 'item' + str(itemId) + ' ' + str(counter) + '\n'
    entity_dict['item'+str(itemId)] = counter
    counter += 1

for column in sideinfo_columns:
    for value in out[column].unique():
        if value == 'NULL':
            continue
        if entity_dict.get(column+str(value)) is None:
            entity_list += column + \
                str(value) + ' ' + str(counter) + '\n'
            entity_dict[column+str(value)] = counter
            counter += 1
f = open("entity_list.txt", "w")
f.write(entity_list)
f.close()

# make knowledge graph file
kg_final = ''
for itemId in out[item_id_column].unique():
    item_entity_id = str(entity_dict['item'+str(itemId)])
    row = out.loc[out[item_id_column] == itemId]

    for index, column in enumerate(sideinfo_columns):
        value = row[column].values[0]
        if value != 'NULL':
            column_entity_id = entity_dict[column+str(value)]
            kg_final += item_entity_id + ' ' + \
                str(index) + ' ' + str(column_entity_id) + '\n'
f = open("kg_final.txt", "w")
f.write(kg_final)
f.close()

relation_list = ''
for index, column in enumerate(sideinfo_columns):
    relation_list +=  column + ' ' + str(index) + '\n'
f = open("relation_list.txt", "w")
f.write(relation_list)
f.close()

train_interaction_dict = {}
train_output = ''
for index, row in train.iterrows():
    userId = user_dict[row[user_id_column]]
    itemId = item_dict.get(row[item_id_column])
    if itemId is None:
        continue
    if train_interaction_dict.get(userId) is None:
        item_set = set()
        train_interaction_dict[userId] = item_set
        train_interaction_dict[userId].add(itemId)
    else:
        train_interaction_dict[userId].add(itemId)
for userId, itemIds in train_interaction_dict.items():
    train_output += str(userId)
    for itemId in itemIds:
        train_output += ' ' + str(itemId)
    train_output += '\n'
f = open("train.txt", "w")
f.write(train_output)
f.close()

test_interaction_dict = {}
test_output = ''
for index, row in test.iterrows():
    userId = user_dict[row[user_id_column]]
    itemId = item_dict.get(row[item_id_column])
    if itemId is None:
        continue
    if test_interaction_dict.get(userId) is None:
        item_set = set()
        test_interaction_dict[userId] = item_set
        test_interaction_dict[userId].add(itemId)
    else:
        test_interaction_dict[userId].add(itemId)
for userId, itemIds in test_interaction_dict.items():
    test_output += str(userId)
    for itemId in itemIds:
        test_output += ' ' + str(itemId)
    test_output += '\n'
f = open("test.txt", "w")
f.write(test_output)
f.close()
