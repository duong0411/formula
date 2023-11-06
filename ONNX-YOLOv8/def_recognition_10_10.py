import os
import sys
import cv2
import numpy as np
import math
import time
import re
import onnxruntime as ort

class BaseRecLabelDecode(object):
    def __init__(self, character_dict_path=None, use_space_char=False):
        self.beg_str = "sos"
        self.end_str = "eos"
        self.reverse = False
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
            if 'arabic' in character_dict_path:
                self.reverse = True

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def pred_reverse(self, pred):
        pred_re = []
        c_current = ''
        for c in pred:
            if not bool(re.search('[a-zA-Z0-9 :*./%+-]', c)):
                if c_current != '':
                    pred_re.append(c_current)
                pred_re.append(c)
                c_current = ''
            else:
                c_current += c
        if c_current != '':
            pred_re.append(c_current)

        return ''.join(pred_re[::-1])

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                    batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[text_id]
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)

            if self.reverse:  # for arabic rec
                text = self.pred_reverse(text)

            result_list.append((text, np.mean(conf_list).tolist()))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank

class CTCLabelDecode(BaseRecLabelDecode):

    def __init__(self, character_dict_path=None, use_space_char=False,
                 **kwargs):
        super(CTCLabelDecode, self).__init__(character_dict_path,
                                             use_space_char)

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[-1]
        # if isinstance(preds, paddle.Tensor):
        #     preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character

class TextRecognizer(object):
    def __init__(self, rec_model_dir, rec_char_dict_path):
        
        self.rec_image_shape = [3, 48, 320]
        self.rec_batch_num = 6
        self.rec_model_dir= rec_model_dir
        self.rec_char_dict_path = rec_char_dict_path

        postprocess_params = {
            "character_dict_path": self.rec_char_dict_path,
            "use_space_char": True
        }
        
        self.postprocess_op = CTCLabelDecode(**postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, self.config = create_predictor(self.rec_model_dir)
        
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        
        # if self.use_onnx:
        w = self.input_tensor.shape[3:][0]
        if isinstance(w, str):
            pass
        elif w is not None and w > 0:
            imgW = w

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list):
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [['', 0.0]] * img_num
        batch_num = self.rec_batch_num
        st = time.time()

        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []

            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH

            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                    
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)
                    
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
                
            # if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = norm_img_batch
            outputs = self.predictor.run(self.output_tensors,
                                            input_dict)
            preds = outputs[0]
            # print(preds)
                        
            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

        return rec_res, time.time() - st

def create_predictor(rec_model_dir):
    model_dir = rec_model_dir

    if model_dir is None:
        sys.exit(0)
        
    model_file_path = model_dir
    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))
    sess = ort.InferenceSession(model_file_path)
    return sess, sess.get_inputs()[0], None, None

def check_and_read(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True, False
    
    elif os.path.basename(img_path)[-3:] in ['pdf']:
        import fitz
        from PIL import Image
        imgs = []
        with fitz.open(img_path) as pdf:
            for pg in range(0, pdf.pageCount):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.getPixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
                
            return imgs, False, True
    return None, False, False

def _check_image_file(path):
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    return any([path.lower().endswith(e) for e in img_end])

def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))
    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'pdf'}
    if os.path.isfile(img_file) and _check_image_file(img_file):
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and _check_image_file(file_path):
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists

def find_exponent_pos(img):
    # img = gray_image[ 1900:1940, 700:1500]
    # img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_test = img.copy()
    try:
        if len(img.shape) == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except: print(img)
    
    denoised_image = cv2.medianBlur(img, 3)
    _, thresh = cv2.threshold(denoised_image, 200, 255, cv2.THRESH_BINARY_INV)

    indices = np.where(thresh == 255)
    xmin, ymin, xmax, ymax = min(indices[1]),  min(indices[0]), max(indices[1]),  max(indices[0]) # find bounding box with min area
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # cv2.rectangle(img_test, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    
    bbox_list = []
    for contour in contours:
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bbox_list.append([x, y, w, h])
        # cv2.rectangle(img_test, (x, y, w, h), (0, 255, 0), 1)

    bbox_array = np.array(bbox_list)
    sorted_array = bbox_array[np.argsort(bbox_array[:,0])]
    sorted_array_ = sorted_array.copy()
    sorted_array_[:,2] =  sorted_array[:,0] + sorted_array[:,2]
    sorted_array_[:,3] =  sorted_array[:,1] + sorted_array[:,3]
    # 
    # print(sorted_array_)
    
    mask = (sorted_array_[:,1] < ymin + 5) & (abs(sorted_array_[:,3]-sorted_array_[:,1]) > (abs(ymax-ymin)*0.33)) & \
            (abs(sorted_array_[:,3]-sorted_array_[:,1]) < abs(ymax-ymin)*0.66) # filter exponent symbols
            
            
    # cv2.imshow("Hi", img_test)
    # cv2.waitKey()
    
    if True not in mask:
        return 0
    else:
        cut_pos_list = []
        list_merge = []
        for i in range(len(sorted_array_)-1): # concate component symbol if it is consequently order
            if (mask[i] == True and mask[i+1] == False): # if the exponent symbol be alone
                if len(list_merge) > 0:
                    list_merge = np.array(list_merge)
                    cut_pos_list.append([min(list_merge[:,0]), min(list_merge[:,1]), max(list_merge[:,2]), max(list_merge[:,3])])
                    list_merge = []
                else: cut_pos_list.append(sorted_array_[i].tolist())
                
            elif (mask[i] == True and mask[i+1] == True): # if the exponent symbol be not alone
                list_merge.append(sorted_array_[i].tolist())
                list_merge.append(sorted_array_[i+1].tolist())
                if i == len(sorted_array_)-2: 
                    list_merge = np.array(list_merge)
                    cut_pos_list.append([min(list_merge[:,0]), min(list_merge[:,1]), max(list_merge[:,2]), max(list_merge[:,3])])
                    
            elif (mask[i] == False and mask[i+1] == True) and i == len(sorted_array_)-2: # if the exponet symbol be last order
                cut_pos_list.append(sorted_array_[i+1].tolist())

    # for i in cut_pos_list:
    #     cv2.rectangle(img_test, (i[0], i[1]), (i[2], i[3]), (0, 255, 255), 1)
    # cv2.imshow("Hi", img_test)
    # cv2.waitKey()
    return cut_pos_list

def split_exponent_img(img_list):
    flag_split_exp = []
    img_list_out = []
    for index_img_list, img in enumerate(img_list):
        
        x1, y1, (y2, x2) = 0, 0 , img.shape[0:2]
        cut_pos_list = find_exponent_pos(img)
        
        if cut_pos_list == 0:
            flag_split_exp.append(0)
            img_list_out.append(img)
        else:
            input_list = []
            # img_list.remove(img)
            for index, i in enumerate(cut_pos_list): # split areas to extract from PDF
                if index == 0: #first id
                    input_list.append([x1, x1 + i[0]-1, y1, y2])
                    input_list.append([x1 + i[0]-1, x1 + i[2], y1, y2, ''])
                elif index == len(cut_pos_list) - 1: #last id
                    input_list.append([x1 + cut_pos_list[index-1][2], x1 + i[0]-1, y1, y2])
                    input_list.append([x1 + i[0]-1, x1 + i[2], y1, y2, ''])
                    input_list.append([x1 + i[2], x2, y1, y2])
                else:
                    input_list.append([x1 + cut_pos_list[index-1][2], x1 + i[0]-1, y1, y2])
                    input_list.append([x1 + i[0]-1, x1 + i[2], y1, y2, ''])
                
                if len(cut_pos_list) == 1: input_list.append([x1 + i[2], x2, y1, y2]) #if cut_pos_list only contain a element.
                
            for index_input_list, j in enumerate(input_list):
                
                #padding
                img_pad = cv2.copyMakeBorder(img[j[2]: j[3], j[0]: j[1]], 1, 1, int(y2*0.25), int(y2*0.25), cv2.BORDER_CONSTANT, value=(255,255,255))
                img_list_out.append(img_pad)
                # cv2.imshow("hi", img_pad)
                # cv2.waitKey()
                if index_input_list == 0: flag_split_exp.append(0) # normal image
                elif len(j) == 5:         flag_split_exp.append(2) # exponent image
                else:                     flag_split_exp.append(1) # sub_image
         
    # for i in img_list_out:
    #     print(i)
    #     cv2.imshow("H", i)
    #     cv2.waitKey()       
    return img_list_out, flag_split_exp
        
def main(img_list, rec_model_dir, rec_char_dict_path):
    
    # image_file_list = get_image_file_list(image_dir)
    text_recognizer = TextRecognizer(rec_model_dir, rec_char_dict_path)
    # valid_image_file_list = []
    # img_list = []

    # for image_file in image_file_list:
    #     img, flag, _ = check_and_read(image_file)
    #     if not flag:
    #         img = cv2.imread(image_file)
    #     if img is None:
    #         continue
    #     valid_image_file_list.append(image_file)
    #     img_list.append(img)
    # cv2.imshow("hi", img_list[0])
    # cv2.waitKey()    
    
    img_list,  flag_split_exp= split_exponent_img(img_list)
    
    # print(img_list[1])
    # for i in img_list:
    #     cv2.imshow("H", i)
    #     cv2.waitKey()
    
    try:
        rec_res, _ = text_recognizer(img_list)

    except Exception as E:
        print("Erorr")
        exit()
        
    rec_res_filter = [i[0] if flag_split_exp[m] != 2 else "^("+i[0]+")" for m, i in enumerate(rec_res)] # add ^( exponent )
    for i in range(len(flag_split_exp)-1, 0, -1):
        if flag_split_exp[i] != 0:
            rec_res_filter[i-1] += rec_res_filter[i]
            rec_res_filter.pop(i)
        # elif flag_split_exp[i] == 2:
            # rec_res_filter[i-1] += "^(" + rec_res_filter[i] + ")"
            # rec_res_filter.pop(i)
        else: pass
        
    out_list = []
    for ino in range(len(rec_res_filter)):
        text_body = "".join(rec_res_filter[ino])
        text_body = re.sub(r"\s", "", text_body)
        text_body = text_body.split('=')
        out_list.append(text_body)
        
        # print(rec_res[ino][0])
    return out_list

a1= time.time()

def Recognition(image_dir, rec_model_dir, rec_char_dict_path):

    return(main(image_dir, rec_model_dir, rec_char_dict_path))
    
if __name__ == "__main__":  
    image_dir = r"/result1"

    image_list = []
    for i in os.listdir(image_dir):
        img_path = os.path.join(image_dir, i)
        #print(img_path)
        image_list.append(cv2.imread(img_path))
    rec_model_dir = r"/rec_10_2.onnx"
    rec_char_dict_path = r"/dict_9_29.txt"
    print(Recognition(image_list, rec_model_dir, rec_char_dict_path), time.time() - a1)
