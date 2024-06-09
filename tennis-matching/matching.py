import argparse
import json
import time
import datetime
from tqdm import tqdm
from bson.objectid import ObjectId
import copy

import pymongo
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.models import GeoRadius, GeoPoint, DatetimeRange, Range, NestedCondition, Nested, FilterSelector

from sentence_transformers import SentenceTransformer

NUM_SECONDS_IN_A_MIN = 60

def read_json(json_path):
    with open(json_path, "r") as f:
        json_result = json.load(f)
    return json_result

def set_mongoDB_client(mongoDB_uri, mongoDB_port_num):
    client = pymongo.MongoClient(mongoDB_uri, port=mongoDB_port_num)

    return client

def set_Qdrant_client(Qdrant_endpoint, Qdrant_api_key):
    return QdrantClient(Qdrant_endpoint, api_key=Qdrant_api_key)

def create_Qdrant_collections(qdrant_client):
    qdrant_client.create_collection(
        collection_name="match_infos",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    qdrant_client.create_collection(
        collection_name="court_infos",
        vectors_config=VectorParams(size=2, distance=Distance.EUCLID),
    )

    return qdrant_client

def delete_Qdrant_collections(qdrant_client):
    qdrant_client.delete_collection(collection_name="match_infos")
    qdrant_client.delete_collection(collection_name="court_infos")

    return qdrant_client

def get_sentence_transformer(sentence_transformer_name):
    sentence_embedding_model = SentenceTransformer(sentence_transformer_name)
    return sentence_embedding_model

def get_Qdrant_match_infos_length(qdrant_client):
    return qdrant_client.count(
        collection_name="match_infos",
        exact=True,
    ).count

def upsert_to_Qdrant_match_infos(mongoDB_client, qdrant_client, sentence_embedding_model, pipeline, _id = None):
    global NUM_SECONDS_IN_A_MIN
    temp_pipeline = copy.deepcopy(pipeline)
    if _id:
        temp_pipeline.append({
        "$match": {
            "_id": _id
        }
        })

    points = []
    for i, post in enumerate(mongoDB_client.tennis.matchRequest.aggregate(temp_pipeline)):
        # print("-----------------------------------")
        # print(post)
        # test = "_id"
        # if test in post:
        #     print(type(post[test]))

        if "dislikedCourts" not in post:
            post["dislikedCourts"] = []

        if "maxDistance" not in post:
            post["maxDistance"] = 4.5

        if "description" not in post:
            post["description"] = ""

        if "minTime" not in post:
            post["minTime"] = 30
        
        if "maxTime" not in post:
            post["maxTime"] = 120

        if post["isReserved"]:
            post["maxDistance"] = 10.0
            post["minTime"] = (post["startTime"] - post["endTime"]).total_seconds() / NUM_SECONDS_IN_A_MIN
            post["maxTime"] = (post["startTime"] - post["endTime"]).total_seconds() / NUM_SECONDS_IN_A_MIN
        else:
            post["reservationCourtId"] = ""
            post["reservationDate"] = ""

        cur_id = get_Qdrant_match_infos_length(qdrant_client)
        points.append(PointStruct(id = cur_id + 1 + i, vector=sentence_embedding_model.encode(post["description"]), payload={
            "_id": post["_id"],
            "userId": post["userId"],
            "location": {"lon": post["location"]["x"], "lat": post["location"]["y"]},
            "startTime": post["startTime"],
            "endTime": post["endTime"],
            "isSingles": post["isSingles"],
            "objective": post["objective"],
            "maxDistance": post["maxDistance"],
            "dislikedCourts": post["dislikedCourts"],
            "minTime": post["minTime"],
            "maxTime": post["maxTime"],
            "description": post["description"],
            "user_ntrp": post["user_ntrp"],
            "isReserved": post["isReserved"],
            "reservationCourtId": post["reservationCourtId"],
            "reservationDate": post["reservationDate"]
        }))

    operation_info = qdrant_client.upsert(
        collection_name="match_infos",
        wait=True,
        points=points
    )
    print(operation_info)
    del temp_pipeline

def delete_from_Qdrant_match_infos(qdrant_client, collection, _id):
    key = ""
    if collection == "user":
        key = "userId"
    elif collection == "matchRequest":
        key = "_id"
    else:
        print("Wrong collection is inputted for deletion of Qdrant")
        return None

    qdrant_client.delete(
        collection_name="match_infos",
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=_id),
                    ),
                ],
            )
        ),
    )
    
def get_Qdrant_court_infos_length(qdrant_client):
    return qdrant_client.count(
        collection_name="court_infos",
        exact=True,
    ).count

def upsert_to_Qdrant_court_infos(mongoDB_client, qdrant_client, pipeline, _id = None):
    temp_pipeline = copy.deepcopy(pipeline)
    if _id:
        temp_pipeline.append({
        "$match": {
            "_id": _id
        }
        })
        
    points = []

    for i, post in enumerate(mongoDB_client.tennis.court.aggregate(temp_pipeline)):
        cur_id = get_Qdrant_court_infos_length(qdrant_client) + 1
        points.append(PointStruct(id = cur_id + 1 +i, vector=post["location"].values(), payload={
            "_id": post["_id"],
            "description": post["description"],
            "courtName": post["courtName"],
            "timeSlotsList": post["timeSlotsList"],
            "courtType": post["courtType"]
        }))

    operation_info = qdrant_client.upsert(
        collection_name="court_infos",
        wait=True,
        points=points
    )

    print(operation_info)
    del temp_pipeline

def delete_from_Qdrant_court_infos(qdrant_client, _id):
    qdrant_client.delete(
        collection_name="court_infos",
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="_id",
                        match=MatchValue(value=_id),
                    ),
                ],
            )
        ),
    )

def modify_timeslots_in_Qdrant_court_infos(mongoDB_client, qdrant_client, _id):
    modified_entity = mongoDB_client.tennis.matchResult.find_one({"_id": ObjectId(_id)})
    courtId = modified_entity["matchDetails"]["courtId"]
    startTime, endTime = modified_entity["matchDetails"]["startTime"], modified_entity["matchDetails"]["endTime"]
    date = startTime.strftime('%Y-%m-%d')

    test_timeslots = []

    cur_min = 0
    while startTime + datetime.timedelta(minutes=cur_min) < endTime + datetime.timedelta(minutes=30):
        test_timeslots.append((startTime + datetime.timedelta(minutes=(cur_min))).strftime('%Y-%m-%dT%H:%M:%S'))
        cur_min += 30

    search_result = qdrant_client.scroll(
        collection_name="court_infos",
        scroll_filter=Filter(
            must=[
                FieldCondition(key="_id", match=MatchValue(value=courtId)),
            ]
        ),
        limit=50,
        with_payload=True,
        with_vectors=False,
    )

    new_timeSlots = dict(search_result[0][0])["payload"]["timeSlotsList"]

    for i, item in enumerate(new_timeSlots):
        if item["date"] == date:
            for j, subitem in enumerate(item["timeSlots"]):
                if subitem["startTime"] in test_timeslots:
                    new_timeSlots[i]["timeSlots"][j]["status"] = "PENDING"
            break

    qdrant_client.set_payload(
        collection_name="court_infos",
        payload={
            "timeSlotsList": new_timeSlots
        },
        points=Filter(
            must=[
                FieldCondition(key="_id", match=MatchValue(value=courtId)),
            ],
        ),
    )

def is_match_request_data_valid(match_request_data):
    # essential_key_list = ["_id", "userId", "location", "startTime", "endTime", "isSingles", "objective", "maxDistance", "dislikedCourts", "minTime", "maxTime", "description", "user_ntrp"]
    essential_key_list = ["_id", "userId", "location", "startTime", "endTime", "isSingles", "objective", "description", "user_ntrp"]
    
    for item in essential_key_list:
        if item not in match_request_data:
            return False
        
    return True

def get_objective_should_list(objective):
    objective_should_list = []
    if objective == "ANY":
        objective_should_list = [
            FieldCondition(
                    key = "objective",
                    match = MatchValue(value = "FUN"),
            ),
            FieldCondition(
                    key = "objective",
                    match = MatchValue(value = "INTENSE"),
            ),
            FieldCondition(
                    key = "objective",
                    match = MatchValue(value = "ANY"),
            )
        ]
    else:
        objective_should_list = [
            FieldCondition(
                    key = "objective",
                    match = MatchValue(value = objective),
            ),
            FieldCondition(
                    key = "objective",
                    match = MatchValue(value = "ANY"),
            )
        ]

    return objective_should_list

def match_request_qdrant_search(qdrant_client, sentence_embedding_model, cur_input):
    global NUM_SECONDS_IN_A_MIN

    must_not_input = [
         FieldCondition(
            key = "userId",
            match = MatchValue(value = cur_input["userId"]),
        )
    ]
    if cur_input["isReserved"]:
        must_not_input.append(
            FieldCondition(
                key = "isReserved",
                match = MatchValue(value = True),
            )
        )

    if "maxTime" not in cur_input:
        cur_input["maxTime"] = (cur_input["endTime"] - cur_input["startTime"]).total_seconds() / NUM_SECONDS_IN_A_MIN

    if "minTime" not in cur_input:
        cur_input["minTime"] = (cur_input["endTime"] - cur_input["startTime"]).total_seconds() / NUM_SECONDS_IN_A_MIN

    if not "maxDistance" in cur_input:
        cur_input["maxDistance"] = 10.0

    search_result = qdrant_client.search(
        collection_name="match_infos",
        query_vector=sentence_embedding_model.encode(cur_input['description']),
        limit=50,
        with_vectors=False,
        with_payload=True,        
        query_filter = Filter(
            must = [
                FieldCondition(
                    key = "isSingles",
                    match = MatchValue(value = cur_input["isSingles"]),
                ),
                FieldCondition(
                    key = "startTime",
                    range=DatetimeRange(
                        gt = None,
                        gte = cur_input["startTime"],
                        lt = None,
                        lte = cur_input["endTime"],
                    ),
                ),
                FieldCondition(
                    key = "endTime",
                    range=DatetimeRange(
                        gt = None,
                        gte = cur_input["startTime"],
                        lt = None,
                        lte = cur_input["endTime"],
                    ),
                ),
                FieldCondition(
                    key="minTime",
                    range=Range(
                        gt = None,
                        gte = None,
                        lt = None,
                        lte = cur_input["maxTime"],
                    ),
                ),
                FieldCondition(
                    key="maxTime",
                    range=Range(
                        gt = None,
                        gte = cur_input["minTime"],
                        lt = None,
                        lte = None,
                    ),
                ),
                FieldCondition(
                    key="location",
                    geo_radius = GeoRadius(
                        center = GeoPoint(
                            lon = cur_input["location"]["x"],
                            lat = cur_input["location"]["y"],
                        ),
                        radius = cur_input["maxDistance"] * 1000,
                    ),
                ),
            ],
            must_not = must_not_input,
            should = get_objective_should_list(cur_input["objective"])
        )
    )

    return search_result

def court_qdrant_search(qdrant_client, cur_input, must_sample):
    search_result = qdrant_client.search(
        collection_name="court_infos",
        query_vector=[cur_input["location"]["x"], cur_input["location"]["y"]],
        limit=1,
        with_vectors=True,
        with_payload=True,
        query_filter = Filter(
            must=[
                NestedCondition(
                    nested=Nested(
                        key="timeSlotsList",
                        filter=Filter(
                            must=must_sample
                        ),
                    )
                )
            ],
            must_not=[
                FieldCondition(key="_id", match=MatchValue(value=dislikedCourt))
            for dislikedCourt in cur_input["disliked_courts"]]
        )
    )
    return search_result

def do_match_request_and_insert_to_mongoDB(mongoDB_client, qdrant_client, sentence_embedding_model, pipeline, _id = None):
    temp_pipeline = copy.deepcopy(pipeline)
    if _id:
        temp_pipeline.append({
        "$match": {
            "_id": _id
        }
        })

    for item in mongoDB_client.tennis.matchRequest.aggregate(temp_pipeline):
        if not is_match_request_data_valid(item):
            print(f"Match request is not valid for {item['_id']}")
            continue

        match_request_search_result = match_request_qdrant_search(qdrant_client, sentence_embedding_model, item)
        if not match_request_search_result:
            print(f"No proper match request result for {item['_id']}")
            continue

        match_request_search_result = sorted(match_request_search_result, key = lambda x: abs(dict(x)["payload"]["user_ntrp"] - item["user_ntrp"]))

        for subitem in tqdm(match_request_search_result):
            result_startTime = datetime.datetime.strptime(dict(subitem)["payload"]["startTime"],  "%Y-%m-%dT%H:%M:%S")
            result_endTime = datetime.datetime.strptime(dict(subitem)["payload"]["endTime"],  "%Y-%m-%dT%H:%M:%S")
            result_maxTime = dict(subitem)["payload"]["maxTime"]
            result_minTime = dict(subitem)["payload"]["minTime"]

            target_startTime = item["startTime"]
            target_endTime = item["endTime"]
            target_maxTime = item["maxTime"]
            target_minTime = item["minTime"]
            
            test_dislikedCourts = []
            if "dislikedCourts" in item:
                test_dislikedCourts += item["dislikedCourts"]
            if "dislikedCourts" in dict(subitem)["payload"]:
                test_dislikedCourts += dict(subitem)["payload"]["dislikedCourts"]

            if item["isReserved"]:
                court_result = {
                    "vector": list(mongoDB_client.tennis.court.find_one({"_id": item["reservationCourtId"]})["location"].values()),
                    "payload": {
                        "_id": item["reservationCourtId"],
                        "courtType": mongoDB_client.tennis.court.find_one({"_id": item["reservationCourtId"]})["courtType"]
                    }
                }

            elif dict(subitem)["payload"]["isReserved"]:
                court_result = {
                    "vector": list(mongoDB_client.tennis.court.find_one({"_id": dict(subitem)["payload"]["reservationCourtId"]})["location"].values()),
                    "payload": {
                        "_id": dict(subitem)["payload"]["reservationCourtId"],
                        "courtType": mongoDB_client.tennis.court.find_one({"_id": dict(subitem)["payload"]["reservationCourtId"]})["courtType"]
                    }
                }

            else:
                global NUM_SECONDS_IN_A_MIN

                final_startTime = target_startTime if (target_startTime > result_startTime) else result_startTime
                temp_endTime = result_endTime if (target_endTime > result_endTime) else target_endTime
                # temp_endTime += datetime.timedelta(minutes=30)
                final_minTime = target_minTime if (target_minTime > result_minTime) else result_minTime
                final_maxTime = result_maxTime if (target_maxTime > result_maxTime) else target_maxTime
                if (temp_endTime - final_startTime).total_seconds() / NUM_SECONDS_IN_A_MIN >= final_maxTime:
                    final_endTime = final_startTime +  datetime.timedelta(minutes=final_maxTime)
                elif (temp_endTime - final_startTime).total_seconds() / NUM_SECONDS_IN_A_MIN < final_minTime:
                    continue
                else:
                    final_endTime = temp_endTime

                result_location = dict(subitem)["payload"]["location"]
                test_location = {"x": (result_location["lon"] + item["location"]["x"]) / 2, "y": (result_location["lat"] + item["location"]["y"]) / 2}

                final_input = {
                    "location": test_location,
                    "startTime" : final_startTime,
                    "endTime" : final_endTime,
                    "disliked_courts": test_dislikedCourts
                }

                court_result = []
                temp_time_add = 0
                while not court_result and final_input["endTime"] + datetime.timedelta(minutes=temp_time_add) < datetime.datetime.combine(final_input["startTime"], datetime.time.max):
                    test_timeslots = []
                    cur_min = 0
                    while final_input["startTime"] + datetime.timedelta(minutes=cur_min) < final_input["endTime"]:
                        test_timeslots.append((final_input["startTime"] + datetime.timedelta(minutes=(cur_min+temp_time_add))).strftime('%Y-%m-%dT%H:%M:%S'))
                        cur_min += 30
                    
                    must_sample = []
                    for time_str in test_timeslots:
                        must_sample.append(FieldCondition(
                                                    key="timeSlots[].startTime", match=MatchValue(value=time_str)
                                                ))
                        must_sample.append(FieldCondition(
                                                    key="timeSlots[].status", match=MatchValue(value="BEFORE")
                                                ))
                        
                    court_result = court_qdrant_search(qdrant_client, final_input, must_sample)
                    if not court_result:
                        temp_time_add += 30
                if court_result:
                    court_result = dict(court_result[0])

            if court_result:
                if {dict(subitem)["payload"]["userId"]: dict(subitem)["payload"]["_id"], item["userId"]: item["_id"]} not in [item["userAndMatchRequests"] for item in mongoDB_client.tennis.matchResult.find()]:
                    sample = {
                        "userAndMatchRequests" :  {item["userId"]: item["_id"], dict(subitem)["payload"]["userId"]: dict(subitem)["payload"]["_id"]},
                        "matchDetails" : {
                            "startTime" : final_input["startTime"] + datetime.timedelta(minutes=temp_time_add),
                            "endTime" : final_input["endTime"] + datetime.timedelta(minutes=temp_time_add),
                            "location" : {"x": court_result["vector"][0], "y": court_result["vector"][1]},
                            "courtId" : court_result["payload"]["_id"],
                            "courtType": court_result["payload"]["courtType"],
                            "isSingles": item["isSingles"],
                            "objective": item["objective"]
                        },
                        "isConfirmed": False,
                        "_class" : "joluphosoin.tennisfunserver.match.data.entity.MatchResult"
                    }
                    mongoDB_client.tennis.matchResult.insert_one(sample)
            else:
                print(f"No proper court for {item['_id']}")
    del temp_pipeline

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_db_urls', type=str, default="jsons/db_urls.json", help="path to db_urls.json")
    parser.add_argument('--path_pipelines', type=str, default="jsons/pipelines.json", help="path to pipelines.json")
    parser.add_argument('--sentence_transformer', type=str, default="snunlp/KR-SBERT-V40K-klueNLI-augSTS", help="sentence transformer model name")
    parser.add_argument('--reset_qdrant_collections', action="store_true", help="only for reset qdrant collections")
    parser.add_argument('--make_current_matchRequests_all', action="store_true", help="reset all and make for all data of current")
    args = parser.parse_args()

    # read mongoDB, Qdrant information json
    db_urls = read_json(args.path_db_urls)

    # set mongoDB client
    mongoDB_client= set_mongoDB_client(db_urls["mongoDB"]["uri"], db_urls["mongoDB"]["port"])

    # get aggregation piepline
    pipelines = read_json(args.path_pipelines)
    match_requests_users_pipeline = pipelines["match_requests_users_pipeline"]
    court_timeslot_pipeline = pipelines["court_timeslot_pipeline"]

    # set Qdrant client
    qdrant_client = set_Qdrant_client(db_urls["Qdrant"]["Endpoint"], db_urls["Qdrant"]["API-KEY"])

    # get sentence transformer
    sentence_embedding_model = get_sentence_transformer(args.sentence_transformer)

    # reset Qdrant client
    if args.reset_qdrant_collections:
        print("reset_qdrant_collections process starts!")
        qdrant_client = delete_Qdrant_collections(qdrant_client)
        qdrant_client = create_Qdrant_collections(qdrant_client)

    elif args.make_current_matchRequests_all:
        print("make_current_matchRequests_all process starts!")
        mongoDB_client.tennis.matchResult.drop()

        qdrant_client = delete_Qdrant_collections(qdrant_client)
        qdrant_client = create_Qdrant_collections(qdrant_client)

        upsert_to_Qdrant_match_infos(mongoDB_client, qdrant_client, sentence_embedding_model, match_requests_users_pipeline)
        upsert_to_Qdrant_court_infos(mongoDB_client, qdrant_client, court_timeslot_pipeline)

        do_match_request_and_insert_to_mongoDB(mongoDB_client, qdrant_client, sentence_embedding_model, match_requests_users_pipeline)

    else:
        print("matching system started!!")
        with mongoDB_client.tennis.watch() as stream:
            while stream.alive:
                change = stream.try_next()
                # print("Current resume token: %r" % (stream.resume_token,))
                if change is not None:
                    try:
                        print("Change document: %r" % (change,))
                        operationType, changed_collection = change["operationType"], change['ns']['coll']
                        if operationType == "insert":
                            if changed_collection == "matchRequest":
                                changed_id = change["fullDocument"]["_id"]
                                upsert_to_Qdrant_match_infos(mongoDB_client, qdrant_client, sentence_embedding_model, match_requests_users_pipeline, changed_id)

                                do_match_request_and_insert_to_mongoDB(mongoDB_client, qdrant_client, sentence_embedding_model, match_requests_users_pipeline)
                                print("*********************************")
                                print("insert matchRequest and making result completed")
                                print("*********************************")

                            elif changed_collection == "timeslot":
                                changed_id = change["fullDocument"]["courtId"]
                                upsert_to_Qdrant_court_infos(mongoDB_client, qdrant_client, court_timeslot_pipeline, changed_id)

                                print("*********************************")
                                print("insert timeslot completed")
                                print("*********************************")
                            else:
                                print(f"MongoDB collection {changed_collection} is inserted, but I don't feel like to do something...")
                        elif operationType == "replace":
                            if changed_collection == "matchResult":
                                changed_id = change["fullDocument"]["_id"]
                                if change["fullDocument"]["isConfirmed"]:
                                    modify_timeslots_in_Qdrant_court_infos(mongoDB_client, qdrant_client, changed_id)
                                    print("*********************************")
                                    print("replace completed")
                                    print("*********************************")
                            elif changed_collection == "matchRequest":
                                changed_id = change["fullDocument"]["_id"]
                                delete_from_Qdrant_match_infos(qdrant_client, changed_collection, changed_id)

                                upsert_to_Qdrant_match_infos(mongoDB_client, qdrant_client, sentence_embedding_model, match_requests_users_pipeline, changed_id)

                                do_match_request_and_insert_to_mongoDB(mongoDB_client, qdrant_client, sentence_embedding_model, match_requests_users_pipeline, changed_id)
                                print("*********************************")
                                print("replace matchRequest and making result completed")
                                print("*********************************")
                            else:
                                print(f"MongoDB collection {changed_collection} is modified, but my bed keep calling me...")
                        elif operationType == "delete":
                            if changed_collection in ["matchRequest", "user"]:
                                changed_id = change["documentKey"]["_id"]
                                delete_from_Qdrant_match_infos(qdrant_client, changed_collection, changed_id)
                                print("*********************************")
                                print("delete completed")
                                print("*********************************")
                            elif changed_collection == "court":
                                changed_id = change["documentKey"]["_id"]
                                delete_from_Qdrant_court_infos(qdrant_client, changed_id)
                                print("*********************************")
                                print("delete completed")
                                print("*********************************")
                            else:
                                print(f"Something of MongoDB collection {changed_collection} is deleted, but I'm too tired to do something...")
                        else:
                            print("Something is changed in mongoDB, but I don't care...")
                    except Exception as e:
                        print(f"exception {e} has occured")
                    continue
                time.sleep(10)
    

    