from __future__ import print_function
import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions


natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username='0bceda96-a518-4425-99ae-60f034b16963',
    password='3YFWZ4kFEmeL')


response = natural_language_understanding.analyze(
    text='California Implements Statewide Ban On All .45 ACP Ammo',
    features=Features(entities=EntitiesOptions(), keywords=KeywordsOptions(sentiment=True)))

print(json.dumps(response, indent=2))