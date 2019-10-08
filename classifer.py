import numpy as np
import pandas as pd
import argparse
import sklearn
import random
import json
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Dictionary mapping categories to a number
categories = {
    "hub": 0,
    "motion": 1,
    "plug": 2,
    "health": 3,
    "camera": 4,
    "printer": 5,
    "bulb": 6,
    "smoke": 7,
    "weather": 8,
    "photo": 9,
    "speaker": 10,
    "scale": 11,
    "sleep": 12,
    "phone": 13,
    "computer": 14,
    "router": 15,
    "blank": 16
}
# Hard coded mac address to category dictionary
device_category = {
    # Amazon Echo
    "44:65:0d:56:cc:d3": "hub",
    # Belkin Motion
    "ec:1a:59:83:28:11": "motion",
    # Belkin Switch
    "ec:1a:59:79:f4:89": "plug",
    # Blipcare BP Meter
    "74:6a:89:00:2e:25": "health",
    # Dropcam
    "30:8c:fb:2f:e4:b2": "camera",
    # Nest Dropcam
    "30:8c:fb:b6:ea:45": "camera",
    # HP Printer
    "70:5a:0f:e4:9b:c0": "printer",
    # LIFX
    "d0:73:d5:01:83:08": "bulb",
    # Nest Smoke
    "18:b4:30:25:be:e4": "smoke",
    # Netatmo Cam
    "70:ee:50:18:34:43": "camera",
    # Netamo Weather
    "70:ee:50:03:b8:ac": "weather",
    # Pixstar Photoframe
    "e0:76:d0:33:bb:85": "photo",
    # Samsung Cam
    "00:16:6c:ab:6b:88": "camera",
    # Smart Things
    "d0:52:a8:00:67:5e": "hub",
    # TP-Link Cam
    "f4:f2:6d:93:51:f1": "camera",
    # TP-Link Plug
    "50:c7:bf:00:56:39": "plug",
    # Triby Speaker
    "18:b7:9e:02:20:44": "speaker",
    # Withings baby
    "00:24:e4:11:18:a8": "camera",
    # Withings scale
    "00:24:e4:1b:6f:96": "scale",
    # Withings Sleep
    "00:24:e4:20:28:c6": "sleep",
    # Ihome plug
    "74:c6:3b:29:d7:1d": "plug",
    # Insteon Camera
    "00:62:6e:51:27:2e": "camera",
    # Not listed?
    "e8:ab:fa:19:de:4f": "blank",
    # Samsung Galaxy Tab
    "08:21:ef:3b:fc:e3": "phone",
    # Android Phone
    "40:f3:08:ff:1e:da": "phone",
    # Laptop
    "74:2f:68:81:69:42": "computer",
    # Macbook
    "ac:bc:32:d4:6f:2f": "computer",
    # Android phone2
    "b4:ce:f6:a7:a3:c2": "phone",
    # Iphone
    "d0:a6:37:df:a1:e1": "phone",
    # Macbook/Iphone
    "f4:5c:89:93:cc:85": "phone",
    # TPLink Router Bridge Lan
    "14:cc:20:51:33:ea": "router"
}

mac_device = {
    # Amazon Echo
    "44:65:0d:56:cc:d3": "Amazon Echo",
    # Belkin Motion
    "ec:1a:59:83:28:11": "Belkin Motion",
    # Belkin Switch
    "ec:1a:59:79:f4:89": "Belkin Switch",
    # Blipcare BP Meter
    "74:6a:89:00:2e:25": "Blipcare BP Meter",
    # Dropcam
    "30:8c:fb:2f:e4:b2": "Dropcam",
    # Nest Dropcam
    "30:8c:fb:b6:ea:45": "Nest Dropcam",
    # HP Printer
    "70:5a:0f:e4:9b:c0": "HP Printer",
    # LIFX
    "d0:73:d5:01:83:08": "LIFX",
    # Nest Smoke
    "18:b4:30:25:be:e4": "Nest Smoke",
    # Netatmo Cam
    "70:ee:50:18:34:43": "Netatmo Cam",
    # Netamo Weather
    "70:ee:50:03:b8:ac": "Netamo Weather",
    # Pixstar Photoframe
    "e0:76:d0:33:bb:85": "Pixstar Photoframe",
    # Samsung Cam
    "00:16:6c:ab:6b:88": "Samsung Cam",
    # Smart Things
    "d0:52:a8:00:67:5e": "Smart Things",
    # TP-Link Cam
    "f4:f2:6d:93:51:f1": "TP-Link Cam",
    # TP-Link Plug
    "50:c7:bf:00:56:39": "TP-Link Plug",
    # Triby Speaker
    "18:b7:9e:02:20:44": "Triby Speaker",
    # Withings baby
    "00:24:e4:11:18:a8": "Withings baby",
    # Withings scale
    "00:24:e4:1b:6f:96": "Withings scale",
    # Withings Sleep
    "00:24:e4:20:28:c6": "Withings Sleep",
    # Ihome plug
    "74:c6:3b:29:d7:1d": "Ihome plug",
    # Insteon Camera
    "00:62:6e:51:27:2e": "Insteon Camera",
    # Not listed?
    "e8:ab:fa:19:de:4f": "Not listed?",
    # Samsung Galaxy Tab
    "08:21:ef:3b:fc:e3": "Samsung Galaxy Tab",
    # Android Phone
    "40:f3:08:ff:1e:da": "Android Phone",
    # Laptop
    "74:2f:68:81:69:42": "Laptop",
    # Macbook
    "ac:bc:32:d4:6f:2f": "Macbook",
    # Android phone2
    "b4:ce:f6:a7:a3:c2": "Android phone2",
    # Iphone
    "d0:a6:37:df:a1:e1": "Iphone",
    # Macbook/Iphone
    "f4:5c:89:93:cc:85": "Macbook/Iphone",
    # TPLink Router Bridge Lan
    "14:cc:20:51:33:ea": "TPLink Router Bridge Lan"
}


def mac_to_category(mac_list):
    # Returns a list of category numbers corresponding to the mac_list
    global categories, device_category
    y = []
    for mac in mac_list:
        category = device_category[mac]
        y.append(categories[category])
    return y


def bag_vocab(train_data, classify_data, target_col):
    # Make a vocabulary for the overall bag of words
    # print(train_data)
    data1 = train_data[target_col].drop_duplicates()
    data2 = classify_data[target_col].drop_duplicates()
    data_list = data1.values.tolist() + data2.values.tolist()
    return set(data_list)


def bag_of_words_maker(data_frame, target_col, vocab):
    # Make a corpus (a list of documents)
    # Each document = device, words in document = data
    # Data = pd frame, Target_col = which column of data you want to convert
    device_data = {}
    for index, row in data_frame.iterrows():
        # Make a nested dictionary
        # key1 = device
        # Value1 = {key2 = target_col:value2 = count}
        # Dictionary = {device:{target:count}}
        key = row['device']
        target_data = row[target_col]
        count = row['count']
        if key in device_data:
            if target_data in device_data[key]:
                # Check if the new entry is bigger than the old one
                # Want the bag to represent the final count
                if device_data[key][target_data] < count:
                    device_data[key][target_data] = count
            else:
                device_data[key][target_data] = count
        else:
            device_data[key] = {target_data: count}
    # Create bag of words
    # Array where rows = device, cols = unique words
    # Convert to list to maintain order when creating bag
    bag_of_words = np.zeros((len(device_data), len(vocab)))
    i = 0
    # Device names is the list of rows corresponding to the bag of words
    # I.e. the first element of device_names is the first row of the bag
    device_names = []
    for device in device_data:
        # For each device, add the count of the unique word
        nested_dict = device_data[device]
        device_names.append(device)
        j = 0
        for target in vocab:
            # If the word exists for the device change the count
            # Otherwise leave it at 0 (initialized value so no need to do)
            # Increment j to continue iteration
            if target in nested_dict:
                bag_of_words[i][j] = nested_dict[target]
            j += 1
        i += 1
    return bag_of_words, device_names


def mult_bayes(data_frame, target_col, vocab):
    # Make the proper class labels for the classifier
    bag_of_words, device_names = bag_of_words_maker(data_frame, target_col, vocab)
    y = np.asarray(mac_to_category(device_names))
    # Make and train the classifier
    clf = MultinomialNB()
    clf.fit(bag_of_words, y)
    return clf


def flow_dup_remove(flow_data):
    # Remove duplicate removes from data
    return flow_data.drop_duplicates(['match_flow', 'device'])


def train(flow_data, domain_data, port_data, port_vocab, domain_vocab):
    # Train the data and create the classifiers
    # Get the 2 mult_bayes classifers
    port_clf = mult_bayes(port_data, "port", port_vocab)
    dns_clf = mult_bayes(domain_data, "domain", domain_vocab)

    # Make the overall random forest classifer
    dupless_data = flow_dup_remove(flow_data)
    no_inf_data = dupless_data.drop(dupless_data[dupless_data.avg_flow_rate == np.inf].index)
    num_data = no_inf_data[['volume', 'duration', 'avg_flow_rate']]
    device_data = no_inf_data[['device']]
    # Assume that the bayes classifiers guesses the label correctly
    # For the training phase, this is a good enough assumption
    num_np = num_data.values
    device_col = device_data.values.tolist()
    device_col_flat = [item for sublist in device_col for item in sublist]
    device_list = mac_to_category(device_col_flat)
    device_np = np.asarray(device_list)
    stacked_device = device_np.reshape((-1, 1))
    # Train data columns are: volume, duration, avg_flow, port, dns
    train_np = np.hstack((num_np, stacked_device, stacked_device))
    # print(train_np)
    rand_for_clf = RandomForestClassifier()
    rand_for_clf.fit(train_np, device_np)
    return port_clf, dns_clf, rand_for_clf


def test(flow_data, domain_data, port_data):
    # Test a random device to see if they are classified properly
    # Remove the later comments if you want to generate csvs from
    # the test device dataframe
    global device_category, mac_device
    mac_list = set(flow_data['device'].tolist())
    mac = random.choice(tuple(mac_list))
    device_name = mac_device[mac]
    print("Testing classification on " + device_name)
    flow_dupless = flow_dup_remove(flow_data)
    test_flow = flow_data[flow_data['device'] == mac]
    train_flow = flow_data[flow_data['device'] != mac]
    # test_flow.to_csv(path_or_buf="test_flow.csv", index=False)

    test_port = port_data[port_data['device'] == mac]
    train_port = port_data[port_data['device'] != mac]
    # test_port.to_csv(path_or_buf="test_port.csv", index=False)

    test_dns = domain_data[domain_data['device'] == mac]
    train_dns = domain_data[domain_data['device'] != mac]
    # test_dns.to_csv(path_or_buf="test_dns.csv", index=False)

    # Make vocabs
    port_vocab = bag_vocab(train_port, test_port, "port")
    domain_vocab = bag_vocab(train_dns, test_dns, "domain")
    port_clf, dns_clf, rand_for_clf = train(train_flow, train_dns,
                                train_port, port_vocab, domain_vocab)
    classify(test_flow, test_dns, test_port, port_clf, dns_clf, 
            rand_for_clf, port_vocab, domain_vocab)


def classify(flow_data, domain_data, port_data, port_clf, dns_clf,
 rand_for_clf, port_vocab, domain_vocab):
    # Classify the data using the classifiers we obtained earlier
    # Assumes we are classifying 1 device
    port_bag, _ = bag_of_words_maker(port_data, "port", port_vocab)
    domain_bag, _ = bag_of_words_maker(domain_data, "domain", domain_vocab)

    # Run the bags through the classifier
    port_predict = port_clf.predict(port_bag)
    domain_predict = dns_clf.predict(domain_bag)

    # Attach the predictions to the number data
    dupless_data = flow_dup_remove(flow_data)
    no_inf_data = dupless_data.drop(dupless_data[dupless_data.avg_flow_rate == np.inf].index)
    num_data = no_inf_data[['volume', 'duration', 'avg_flow_rate']]
    num_np = num_data.values
    row_count = int(num_np.shape[0])
    port_predict_list = [port_predict[0]] * row_count
    domain_predict_list = [domain_predict[0]] * row_count
    port_predict_array = np.asarray(port_predict_list)
    domain_predict_array = np.asarray(domain_predict_list)
    port_np = port_predict_array.reshape((-1, 1))
    domain_np = domain_predict_array.reshape((-1, 1))
    classify_np = np.hstack((num_np, port_np, domain_np))
    results_np = rand_for_clf.predict(classify_np)
    results = results_np.tolist()
    classification = max(results, key=results.count)
    # Make a reverse classifaction dictionary
    rev_categories = {}
    for key in categories:
        value = categories[key]
        rev_categories[value] = key
    # Print the final result
    print(rev_categories[classification])


if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description='Create a model to classify IOT devices and test a device')
    parser.add_argument('-f', '--flow', required=True, dest='flow',
                        help='Flow Data in CSV format')
    parser.add_argument('-d', '--domain', required=True, dest='domain',
                        help='Domain data in CSV format')
    parser.add_argument('-p', '--port', required=True, dest='port',
                        help='Port data in CSV format')
    parser.add_argument('-a', '--flow2', required=False, dest='flow_classify',
                        help='Flow data to classify')
    parser.add_argument('-b', '--domain2', required=False, dest='domain_classify',
                        help='Domain data to classify')
    parser.add_argument('-c', '--port2', required=False, dest='port_classify',
                        help='Port data to classify')
    parser.add_argument('-t', '--test', required=False, dest='test',
                        action='store_true', default=False,
                        help='Option to test the classifier using a subset of the training data')
    args = parser.parse_args()

    # Open the proper files and make a pandas dataframe
    # For training data
    flow_data_fn = args.flow
    domain_data_fn = args.domain
    port_data_fn = args.port

    flow_file = open(flow_data_fn, 'r')
    flow_data = pd.read_csv(flow_file)
    flow_file.close()

    domain_file = open(domain_data_fn, 'r')
    domain_data = pd.read_csv(domain_file)
    domain_file.close()

    port_file = open(port_data_fn, 'r')
    port_data = pd.read_csv(port_file)
    port_file.close()

    # Check if we want to classify data or test the classifier
    # We test the classifier via using a random subset of
    # the training data
    if args.test:
        # Test the data, and don't classify the data
        test(flow_data, domain_data, port_data)
        exit()
    else:
        # Open the classifaction data
        flow_classify_fn = args.flow_classify
        domain_data_fn = args.domain_classify
        port_data_fn = args.port_classify

        flow_classify_file = open(flow_classify_fn, 'r')
        flow_classify_data = pd.read_csv(flow_classify_file)
        flow_classify_file.close()

        domain_classify_file = open(domain_data_fn, 'r')
        domain_classify_data = pd.read_csv(domain_classify_file)
        domain_classify_file.close()

        port_classify_file = open(port_data_fn, 'r')
        port_classify_data = pd.read_csv(port_classify_file)
        port_classify_file.close()

        # Set up the vocabs for mult bayes
        port_vocab = bag_vocab(port_data, port_classify_data, "port")
        domain_vocab = bag_vocab(domain_data, domain_classify_data, "domain")

        # Get the three classifiers from training
        port_clf, dns_clf, rand_for_clf = train(flow_data, domain_data, port_data,
        port_vocab, domain_vocab)

        # Classify the data
        classification = classify(flow_classify_data, domain_classify_data,
                port_classify_data, port_clf, dns_clf, rand_for_clf,
                port_vocab, domain_vocab)
