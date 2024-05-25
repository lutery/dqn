'''
1. selenium 模拟长按释放动作执行时间超过0.5秒左右，会导致游戏中的小人跳跃失败
原因是如果执行鼠标移动后在执行长按或者释放的动作会导致执行时间过长，解决办法就是将鼠标移动到中心位置后，
不再移动即可减少按、释放动作执行的时间，通过测试，时间缩短为0.03s左右
'''
from PIL import Image
import io
import time
from enum import Enum
import numpy as np
import threading
import tornado
import tornado.websocket
import tornado.httpserver
import tornado.ioloop
import tornado.web
from tornado.websocket import WebSocketHandler
import json
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
import random


GAME_SHAPE = (512, 768)
CROP_SHAPE = (512, 512)

class JUMP_ACTION(Enum):
    PRESS = (0, 1),
    RELEASE = (1, 0),
    NONE = (0, 0)

web_server_thread = None

class JUMP_STATUS(Enum):
    # // 初始
    init = 1,
    # // 蓄力
    storage = 1<<1,
    # // 跳跃
    jumping = 1<<2,
    # // 当前盒子
    stay = 1<<3,
    # // 下个盒子
    nextBox = 1<<4,
    # // 出界
    outRange = 1<<5,
    # // 当前盒子边缘前向掉落
    current_edge_front = 1<<6,
    # // 下一个盒子边缘后向掉落
    next_edge_back = 1<<7,
    # // 下一个盒子边缘前向掉落
    next_edge_front = 1<<8,
    next_box_centern = 1<<9


def run_wx_jump_game_in_thread(port):
    global web_server_thread
    if web_server_thread is None or not web_server_thread.is_alive():
        web_server_thread = threading.Thread(target=run_tornado, args=(port,))
        web_server_thread.start()

def run_tornado(port):
    web = tornado.web.Application([('/status', WxJumpGame)])
    wb_http_server = tornado.httpserver.HTTPServer(web)
    wb_http_server.listen(port)
    io_loop_instance = tornado.ioloop.IOLoop.current()
    io_loop_instance.start()

def flatten(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten(item))
        elif isinstance(item, dict):
            flattened.extend(item.values())
        else:
            flattened.append(item)
    return flattened

class WxJumpGame(WebSocketHandler):
    clients = set()
    client = None

    def __init__(self, *args) -> None:
        super().__init__(*args)

        self.score = 0
        self.status = 1
        self.done = True
        self.info = None
        self.obs = None

        self.client_socket = None
        self.event = threading.Event()
        # self.async_flag = True


    def open(self):
        if len(WxJumpGame.clients) > 0:
            list(WxJumpGame.clients)[0].close()
        WxJumpGame.clients.add(self)
        WxJumpGame.client = self
        print(f"===> 接受到客户端连接 <===")


    def on_message(self, message):
        # 当客户端连接上时触发
        # 接收客户端发送的消息
        # print(f"===> Received: {message} <===")

        # 处理消息
        # processed_message = f"I got your message: {message}"
        try:
            jump_status = json.loads(message)
            if jump_status["type"] == 'status':
                self.status = jump_status["status"]
                self.score = jump_status["score"]
                self.done = jump_status["done"]
                # self.info = jump_status["info"]
                self.obs = flatten(jump_status["obs"])

            if self.event:
                self.event.set()
        except Exception as e:
            print(f"Error: {e}")

    def on_close(self):
        WxJumpGame.clients.remove(self)

    @staticmethod
    def stop_all():
        for client in WxJumpGame.clients:
            client.close()
        
        WxJumpGame.clients.clear()

    def check_origin(self, origin: str) -> bool:
        return True

    def send_action(self, action):
        # print(f"__send_action: current thread id: {threading.get_ident()}")
        message = {"type": action}
        # print("send message:", message)
        self.write_message(json.dumps(message))
        # print(f"event===> {self.event.is_set() }<===event")
        if not self.event.wait(3.0):
            raise TimeoutException("等待超时，疑似内存崩溃")
        self.event.clear()
        # print("===> send_action end <===")


class WxJump:

    def __init__(self, web_driver_type, port = 8866):
        self.pre_score = 0
        self.pre_action = JUMP_ACTION.NONE
        self.driver = None
        self.driver_type = web_driver_type
        self.port = port

        run_wx_jump_game_in_thread(self.port)
        self.__connect_to_jump_game()

    
    def init_state(self):
        self.close_game()
        self.pre_score = 0
        self.pre_action = JUMP_ACTION.NONE
        run_wx_jump_game_in_thread(self.port)
        self.__connect_to_jump_game()


    def close_game(self):
        if self.driver is not None:
            self.driver.quit()
            self.driver = None

    def print_driver_log(self):
        # print("==== print drviver log start ====")
        # 获取并打印浏览器控制台日志
        logs = self.driver.get_log('browser')
        for log in logs:
            print(log)

        # print("==== print drviver log end ====")

    def __get_driver(self):
        if self.driver_type == "chrome":
            return webdriver.Chrome()
        else:
            return webdriver.Firefox()

    def __connect_to_jump_game(self):
        self.driver = self.__get_driver()
        self.driver.set_window_size(GAME_SHAPE[0], GAME_SHAPE[1])
        self.driver.get("http://localhost:8080")

        try:
            self.move_to_center_action = ActionChains(self.driver)
            element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "statusListenerButton"))
            )
            port_input = self.driver.find_element("id", "port")
            port_input.send_keys(str(self.port))
            print("Page loaded successfully")
            element.click()
            time.sleep(1)
            # 获取窗口尺寸
            size = self.driver.get_window_size()
            self.width = size['width']
            self.height = size['height']
            self.click_pos = (self.width / 2, self.height / 2)
            self.move_to_center_action.move_by_offset(self.click_pos[0], self.click_pos[1])
        except TimeoutException as te:
            print("Page loading took too much time!")
            raise te
            


    def cur_jump_info(self, get_state=True):
        score, is_done, obs = WxJumpGame.client.score, WxJumpGame.client.done, WxJumpGame.client.obs
        game_state = None
        if get_state:
            game_capture_data = self.driver.get_screenshot_as_png()
            png_stream = io.BytesIO(game_capture_data)
            game_capture = Image.open(png_stream)
            left = (game_capture.width - CROP_SHAPE[0]) / 2
            top = (game_capture.height - CROP_SHAPE[1]) / 2
            right = left + CROP_SHAPE[0]
            bottom = top + CROP_SHAPE[1]
            game_capture = game_capture.crop((int(left), int(top), int(right), int(bottom)))
            game_state = np.array(game_capture)


        reward = -3 if is_done else score - self.pre_score
        reward = reward if reward > 0 else (-3 if is_done else -1)
        self.pre_score = score

        return game_state, reward, is_done, obs

    def reset(self, get_state = True):
        # print("===> reset start <===")
        self.pre_score = 0
        self.pre_action = JUMP_ACTION.NONE
        WxJumpGame.client.send_action("reset")
        # print("===> reset end <===")
        return self.cur_jump_info(get_state)

    def step(self, action, get_state = True):
        # print(f"===> {action} step start <===")
        if action[1] == 1 and self.pre_action != JUMP_ACTION.PRESS:
            self.press(self.click_pos)
            self.pre_action = JUMP_ACTION.PRESS
        elif action[1] == 0 and self.pre_action!= JUMP_ACTION.RELEASE:
            self.release(self.click_pos)
            self.pre_action = JUMP_ACTION.RELEASE

        # print(f"===> {action} step end <===")
        return self.cur_jump_info(get_state)
    
    def step_press_up(self, press_time):
        self.step((0, 1), False)
        time.sleep(press_time)
        _, reward, is_done, obs = self.step((1, 0), False)
        if is_done:
            return None, reward, is_done, obs
        else:
            time.sleep(0.7)
            return self.cur_jump_info(True)[0], reward, is_done, obs

    def press(self, pos):
        # print("--------------------------------------------------------")
        self.move_to_center_action.click_and_hold().perform()

    def release(self, pos):
        # print("--------------------------------------------------------")
        # print("===> release start <===")
        self.move_to_center_action.release().perform()
        WxJumpGame.client.send_action("status")
        # print("===> release end <===")


def test_jump_game():
    wxJump = WxJump(web_driver_type="firefox")
    wxJump.reset()
    wxJump.reset()
    wxJump.step_press_up(1.1)
    # wxJump1.reset()
    # wxJump1.step_press_up(1.1)
    # wxJump2.reset()
    # wxJump2.step_press_up(1.1)
    time.sleep(10)
            # print(wxJump.reset()[1:])
            # print(f"reset time: {time.time() - t}")
            # t = time.time()
            # print(wxJump.step((0, 1))[1:])
            # print(f"step time: {time.time() - t}")
            # time.sleep(0.999)
            # t = time.time()
            # print(wxJump.step((1, 0))[1:])
            # print(f"step time: {time.time() - t}")
            # print(wxJump.step((0, 1))[1:])
            # time.sleep(0.3)
            # print(wxJump.step((1, 0))[1:])
            # print(wxJump.reset()[1:])
            # print(wxJump.step((0, 1))[1:])
            # time.sleep(0.03)
            # print(wxJump.step((1, 0))[1:])
            # print(wxJump.step((0, 1))[1:])
            # time.sleep(0.03)
            # print(wxJump.step((1, 0))[1:])
            # print(wxJump.step((0, 1))[1:])
            # time.sleep(0.03)
            # print(wxJump.step((1, 0))[1:])
            # print(wxJump.reset()[1:])
            # print(wxJump.step((0, 1))[1:])
            # time.sleep(0.03)
            # print(wxJump.step((1, 0))[1:])
            # print(wxJump.reset()[1:])
        # except Exception as e:
            # print(f"train error: {e}")
            # wxJump.init_state()
            # time.sleep(100)


    wxJump.close_game()


if __name__ == '__main__':
    test_jump_game()