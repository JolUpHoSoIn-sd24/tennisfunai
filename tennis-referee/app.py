# from flask import Flask
# from flask_restful import Api
# # from flask_cors import CORS, cross_origin
# from flask_restful import Resource
# from flask_restful import reqparse

# import json

# class calibration(Resource):
#     def get(self):
#         # parser = reqparse.RequestParser()
#         # parser.add_argument('url', required=True, type=str, help='url cannot be blank')
#         # args = parser.parse_args()
        
#         jstr = json.dumps({"status": "success"})

#         return jstr

# app = Flask(__name__)
# api = Api(app)
# api.add_resource(calibration, '/')
# # CORS(app)

# if __name__ == '__main__':
#     # app.run(host='34.64.135.47', port=8000, debug=True)
#     app.run(host='0.0.0.0', port=8080, debug=True)

from flask import Flask
from flask_restx import Resource, Api, reqparse
from PIL import Image
import cv2
import numpy as np
from io import BytesIO
import base64

def base64_to_img(base64_str):
    img_out = Image.open(BytesIO(base64.b64decode(base64_str)))
    img_out = np.array(img_out)
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    return img_out

app = Flask(__name__)
api = Api(app)


@api.route('/referee')  
class Hello(Resource):
    def get(self):
        return {"message": "Hiiiiii"}
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('image', required=True, type=str, help='url cannot be blank')
        parser.add_argument('time', required=True, type=str, help='url cannot be blank')
        args = parser.parse_args()
        
        print(f"image: {args.image}")
        cv2.imwrite(f'./test_results/output_{args.time}.png', base64_to_img(args.image))  
        return {"message" : "Welcome, %s!" % args.image}
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)