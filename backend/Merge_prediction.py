import json
import numpy as np
import pandas as pd
'''
Massimo's prediction list:
778528661901234176 -> 1143578752 = 0
786369653421576192 -> 1143578752 = 0
777906418527637504 -> 1182222215 = 0
815991020806635520 -> 639518021 = 0
779051536043442176 -> 953027438 = 0
677836006897266688 -> E17 = 0
660652127258173440 -> E219 = 0
677222986429018112 -> E27 = 1
655136737853853700 -> E294 = 1
639150523304689664 -> E441 = 0
674359085098123265 -> E50 = 1
605394876058992641 -> E558 = 1
625323253930668033 -> E596 = 0
613707919402606592 -> E725 = 0
604303146203103233 -> E782 = 0
'''

massimo_predictions = {'486973039': 1, '566974132': 1, 'E646': 0, 'E219': 1, 'TM927': 0, '1171979372': 0, 'E27': 1, '1607992673': 1, '1849428880': 0, 'E238': 1, 'E211': 1, 'E346': 1, '1644550877': 1, '1041287964': 0, 'E125': 0, '524900646': 0, 'TM1193': 0, 'twittersummize': 1, 'E161': 1, 'E97': 1, 'TM2156': 0, 'E637': 1, 'E385': 1, 'E133': 0, 'E558': 1, '1298683934': 0, 'TM2182': 0, '862182597': 0, 'TM1700': 0, 'E429': 1, '179052119': 0, 'E574': 0, 'E171': 1, 'Dell': 1, 'E289': 1, 'E611': 0, '1195945314': 0, 'TM1789': 0, '1851000424': 1, 'E414': 0, 'E339': 0, '1628281499': 0, '1774176943': 1, '1621413685': 0, 'E222': 0, 'E85': 0, 'TM1326': 0, 'E453': 1, 'E745': 1, '1150437727': 0, 'E313': 0, 'TM1344': 0, '554320262': 0, 'E544': 1, 'E301': 1, '1801700202': 1, 'TM2562': 0, 'TM926': 0, 'E392': 1, 'E764': 0, 'TM1018': 0, 'E22': 1, 'E718': 1, 'TM392': 1, 'E338': 0, 'E739': 0, '1781464158': 0, 'E459': 1, '1201107441': 1, 'TM728': 0, '1040089344': 0, 'E730': 0, 'E502': 0, 'E401': 1, 'E491': 1, '1510762474': 0, '361089316': 0, 'TM1845': 0, 'E61': 1, '1250251349': 0, '1182222215': 0, '1321823288': 1, 'E177': 0, '326860575': 0, '783977636': 1, '824303018': 0, 'E251': 1, 'E675': 1, '378888966': 0, 'E556': 1, 'E237': 1, 'E711': 0, 'E435': 1, 'E92': 1, 'E149': 1, '897082418': 1, '1098530908': 0, '1143976041': 0, '248414896': 0, 'E269': 1, 'E610': 0, 'E182': 1, '682970909': 0, '2049386674': 0, 'E155': 0, 'TM1565': 0, 'E430': 1, '810957746': 0, 'E682': 0, 'E336': 1, '499858722': 0, 'E715': 1, 'E94': 1, 'E322': 1, 'E332': 1, '1325962491': 1, '1785433948': 0, '2029226078': 1, 'E605': 0, 'E691': 1, 'TM1083': 0, 'E670': 0, '710284651': 0, 'ByrdBillings': 1, 'E769': 1, 'E303': 1, 'TM1701': 1, 'ElephantPaint': 0, 'SquareWatermelon': 0, 'Sicko': 0, 'E279': 1, '1017430834': 1, '749948235': 0, 'E720': 1, 'E48': 1, 'E669': 0, 'E294': 1, 'E481': 0, 'E662': 1, 'E591': 1, 'E153': 1, 'E50': 1, 'E679': 1, 'E223': 1, '953027438': 0, 'E499': 0, 'SarahJessicaSurrogate': 1, 'E399': 1, '309522684': 1, 'E17': 0, 'E296': 1, 'E349': 0, 'E631': 1, 'E782': 1, 'E36': 1, '156039748': 0, 'E494': 1, 'TM555': 0, 'E772': 0, 'E348': 1, 'E596': 0, 'E128': 1, '245477424': 0, 'E484': 1, 'E735': 1, '639518021': 0, 'heathLedger': 0, 'E726': 1, 'E47': 1, 'E643': 1, '551279728': 1, 'E111': 1, '154694761': 0, 'E26': 0, '1143578752': 0, 'NikonD300s': 0, '1974749660': 1, 'E725': 0, 'E37': 1, '1729486603': 1, 'E441': 0, 'TM2100': 0, 'E462': 0, 'E540': 1, '958280924': 1, 'E119': 1, 'E493': 0, '2077059694': 1, 'TM659': 0, '758307857': 0, 'E740': 0, '1912098764': 0, '1912650799': 0, '1304370223': 1, 'TM2113': 0, '738161965': 0, 'E737': 0, 'E200': 0, 'E160': 1, 'E137': 1, 'E64': 1, '2111300857': 0, 'E672': 0, 'E659': 1, '946664996': 0, 'E464': 0, '1780898152': 0, 'E657': 1, '437353057': 0, 'E674': 1, 'TM2459': 0, 'E424': 1, 'E235': 0, '925720639': 0, 'E157': 0, '723692095': 0, 'Airfrance': 0, 'TM2605': 0, 'E451': 0, 'E792': 0, 'E40': 1, '1067653684': 0, '1115057873': 1, 'E287': 0, 'E347': 0, 'E696': 1, '855496403': 1, 'E606': 1, 'E312': 0, 'surrogateMom': 0, '1204336037': 1, 'E335': 1, 'TM1341': 0, 'E785': 0, '350312844': 0, 'E753': 0, 'E749': 0, 'E490': 1, '350801120': 0, 'E614': 1, 'E699': 1, 'E470': 1, 'E471': 0, 'TM1708': 0, 'E18': 0, 'E748': 1, '336923905': 0, 'TM2214': 0, 'TM2039': 0, 'E107': 1, '1376241265': 1, 'E683': 1, 'E766': 0, '832438860': 1, '188189965': 1, 'E175': 1, 'E112': 1, '1160390454': 0, 'Vanessa': 1, 'E594': 0, '1093539503': 1, '1375662255': 0, 'E370': 1, 'E546': 1, 'Giantcoconutcrab': 1}
OUTPUT_FILE = "dataset/final_ensemble_prediction.json"

final_predictions = open(OUTPUT_FILE, "w")

with open("dataset/FastText_prediction.json") as fast_text_predictions:
    fast_text_predictions = json.load(fast_text_predictions)

with open("dataset/QDA_prediction_2.json") as QDA_predictions:
    QDA_predictions = json.load(QDA_predictions)

with open("dataset/final_test_dataset_5.csv") as prova:
    prova = pd.read_csv(prova, sep="|")

to_dump = []
correct_predictions = 0
error_predictions = 0
for index in range(len(fast_text_predictions)):

        tweet_id = fast_text_predictions[index]["tweet_id"]
        prediction_id = prova.loc[prova["id"] == int(tweet_id)]["event_id"]
        massimo_prediction = massimo_predictions[prediction_id.tolist()[0]]
        actual_fake = fast_text_predictions[index]["actual_fake"]
        tweet_emotion = QDA_predictions[index]["emotion"]
        fast_text_prediction = fast_text_predictions[index]["predictedId"]
        qda_prediction = QDA_predictions[index]["predictedId"]
        #massimo_prediction = massimo_predictions[int(tweet_id)]
        print(massimo_prediction, fast_text_prediction, qda_prediction)
        final_prediction = int(massimo_prediction) + int(fast_text_prediction) + int(qda_prediction)
        if final_prediction == 0 or final_prediction == 1:
            final_prediction = 0
        else:
            final_prediction = 1
        error = np.abs(int(final_prediction) - int(actual_fake))
        if error == 0:
            correct_predictions += 1
        else:
            error_predictions += 1
        to_dump.append({"tweet_id": tweet_id, "predictedId": final_prediction, "actualFake":actual_fake, "emotion":tweet_emotion})

print(correct_predictions, error_predictions)
json.dump(to_dump, final_predictions)

