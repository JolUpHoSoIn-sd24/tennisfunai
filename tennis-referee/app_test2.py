import argparse
import asyncio
import json
import logging
import os
import ssl
import uuid

import cv2
from aiohttp import web
from av import VideoFrame
import aiohttp_cors
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay

import cv2
from court_detection_net import CourtDetectorNet
import numpy as np
from court_reference import CourtReference
from bounce_detector import BounceDetector
from person_detector import PersonDetector
from ball_detector import BallDetector
from utils import scene_detect
import argparse
import torch
import os
from glob import glob

ROOT = os.path.dirname(__file__)

logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

path_ball_track_model =  "./models/model_best.pt"
path_court_model = "./models/model_tennis_court_det.pt" 
path_bounce_model = "./models/ctb_regr_bounce.cbm"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

print("Loading models...")
ball_detector = BallDetector(path_ball_track_model, device)
court_detector = CourtDetectorNet(path_court_model, device)
person_detector = PersonDetector(device)
bounce_detector = BounceDetector(path_bounce_model)
print("Models loaded!")

x_ball, y_ball = [], []
frame_prev, frame_preprev = None, None
ball_track = [(None, None)]*2

class VideoTransformTrack(MediaStreamTrack):

    kind = "video"

    def __init__(self, track, transform):
        super().__init__()  # don't forget this!
        self.track = track
        self.transform = transform
        self.index = 0
        
    def get_court_img(self):
        court_reference = CourtReference()
        court = court_reference.build_court_reference()
        court = cv2.dilate(court, np.ones((10, 10), dtype=np.uint8))
        court_img = (np.stack((court, court, court), axis=2)*255).astype(np.uint8)
        return court_img
    
    def inf_image(self, frames, draw_trace=True, trace=7):
        global ball_detector, court_detector, person_detector, bounce_detector
        global x_ball, y_ball, frame_prev, frame_preprev, ball_track
        
        ball_track.append(ball_detector.infer_model_for_single_frame(frames[0], frame_prev, frame_preprev, ball_track))
        homography_matrices, kps_court = court_detector.infer_model(frames)
        persons_top, persons_bottom = person_detector.track_players(frames, homography_matrices, filter_players=False)
        
        x_ball.append(ball_track[-1][0])
        y_ball.append(ball_track[-1][1])
        bounces = bounce_detector.predict(x_ball, y_ball) 
        
        try:             
            imgs_res = []
            width_minimap = 166
            height_minimap = 350
            is_track = [x is not None for x in homography_matrices] 
            
            court_img = self.get_court_img()
            
            for i in range(len(frames)):
                img_res = frames[i]
                inv_mat = homography_matrices[i]

                # draw ball trajectory
                if ball_track[i][0]:
                    if draw_trace:
                        for j in range(0, trace):
                            if i-j >= 0:
                                if ball_track[i-j][0]:
                                    draw_x = int(ball_track[i-j][0])
                                    draw_y = int(ball_track[i-j][1])
                                    img_res = cv2.circle(frames[i], (draw_x, draw_y),
                                    radius=3, color=(0, 255, 0), thickness=2)
                    else:    
                        img_res = cv2.circle(img_res , (int(ball_track[i][0]), int(ball_track[i][1])), radius=5,
                                                color=(0, 255, 0), thickness=2)
                        img_res = cv2.putText(img_res, 'ball', 
                                org=(int(ball_track[i][0]) + 8, int(ball_track[i][1]) + 8),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.8,
                                thickness=2,
                                color=(0, 255, 0))

                # draw court keypoints
                if kps_court[i] is not None:
                    for j in range(len(kps_court[i])):
                        img_res = cv2.circle(img_res, (int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])),
                                            radius=0, color=(0, 0, 255), thickness=10)
                        # print(f"i,j: {[i,j]}, coords: {[int(kps_court[i][j][0, 0]), int(kps_court[i][j][0, 1])]}")

                height, width, _ = img_res.shape

                # draw bounce in minimap
                if i in bounces and inv_mat is not None:
                    ball_point = ball_track[i]
                    ball_point = np.array(ball_point, dtype=np.float32).reshape(1, 1, 2)
                    ball_point = cv2.perspectiveTransform(ball_point, inv_mat)
                    court_img = cv2.circle(court_img, (int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])),
                                                        radius=0, color=(0, 255, 255), thickness=50)
                    print(f"bounce_coords: {[int(ball_point[0, 0, 0]), int(ball_point[0, 0, 1])]}")

                minimap = court_img.copy()

                # draw persons
                persons = persons_top[i] + persons_bottom[i]                    
                for j, person in enumerate(persons):
                    if len(person[0]) > 0:
                        person_bbox = list(person[0])
                        img_res = cv2.rectangle(img_res, (int(person_bbox[0]), int(person_bbox[1])),
                                                (int(person_bbox[2]), int(person_bbox[3])), [255, 0, 0], 2)

                        # transmit person point to minimap
                        person_point = list(person[1])
                        person_point = np.array(person_point, dtype=np.float32).reshape(1, 1, 2)
                        person_point = cv2.perspectiveTransform(person_point, inv_mat)
                        minimap = cv2.circle(minimap, (int(person_point[0, 0, 0]), int(person_point[0, 0, 1])),
                                                            radius=0, color=(255, 0, 0), thickness=80)

                minimap = cv2.resize(minimap, (width_minimap, height_minimap))
                img_res[30:(30 + height_minimap), (width - 30 - width_minimap):(width - 30), :] = minimap
                imgs_res.append(img_res)
            
            return imgs_res 
        except Exception as e:
            print(f"No valid image input!")
            print(f"Error: {e}")
            return frames

    async def recv(self):
        frame = await self.track.recv()
        
        if self.transform == "referee":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            # rows, cols, _ = img.shape
            # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            # img = cv2.warpAffine(img, M, (cols, rows))
            
            frames = [img]
            global frame_preprev, frame_prev
            
            if not frame_preprev:
                frame_preprev = img
            if not frame_prev:
                frame_prev = img
            
            img = self.inf_image(frames, draw_trace=True)
            
            del frame_preprev
            frame_preprev = frame_prev
            frame_prev = img

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img[0], format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "calibration":
            pass
        else:
            return frame


async def referee(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pcs.add(pc)

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    recorder = MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        pc.addTrack(
            VideoTransformTrack(
                relay.subscribe(track), transform=params["video_transform"]
            )
        )

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


app = web.Application()
cors = aiohttp_cors.setup(app)
app.on_shutdown.append(on_shutdown)
app.router.add_post("/referee", referee)

for route in list(app.router.routes()):
    cors.add(route, {
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert-file", help="SSL certificate file (for HTTPS)")
    parser.add_argument("--key-file", help="SSL key file (for HTTPS)")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP server (default: 8080)"
    )
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.cert_file:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.cert_file, args.key_file)
    else:
        ssl_context = None

    web.run_app(
        app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context
    )