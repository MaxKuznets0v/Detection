import PySimpleGUI as sg
from PIL import Image
import io
import cv2


def app():
    file_input_column = [
        [sg.Text('Выберите изображение')], [sg.FileBrowse('Найти', file_types=(("PNG files", "*.png"),
        ("JPG files", "*.jpg"), ("ALL files", "*")), key='-load_source-', enable_events=True)],
        [sg.Text('Выберите шаблон')], [sg.FileBrowse('Найти', file_types=(("PNG files", "*.png"),
        ("JPG files", "*.jpg"), ("ALL files", "*")), key='-load_template-', enable_events=True)]]

    layout = [
        [sg.Column(file_input_column, justification='center')],
        [sg.Column([[sg.Image(key='-image-'), sg.Image(key='-template-'), sg.Image(key='-result-')]], justification='center')],
        [sg.Button('Template Matching', key='-detect_tm-'), sg.Button('Viola Jones', key='-detect_vj-')]]

    window = sg.Window('Нахождение лиц', layout, resizable=True, finalize=True)
    while True:
        event, values = window.read()

        if event == '-detect_tm-':
            source_path = values['-load_source-']
            template_path = values['-load_template-']
            if len(source_path) == 0 or len(template_path) == 0:
                continue
            img = cv2.imread(source_path, 1)
            temp = cv2.imread(template_path, 0)
            best = 0
            best_res = 0
            cp = temp.copy()
            for i in range(1, 100):
                factor = 1 + 0.1 * i
                cp = cv2.resize(temp, (int(temp.shape[1] * factor), int(temp.shape[0] * factor)))
                try:
                    res = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cp, cv2.TM_CCOEFF_NORMED)
                except:
                    break
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                w, h = cp.shape[::-1]
                bottom_right = (max_loc[0] + w, max_loc[1] + h)
                if max_val > best:
                    best = max_val
                    best_res = max_loc
                    best_right = (max_loc[0] + w, max_loc[1] + h)
            cnt = 0
            for i in range(1, 100):
                factor = 1 - 0.1 * i
                try:
                    cp = cv2.resize(temp, (int(temp.shape[1] * factor), int(temp.shape[0] * factor)))
                    res = cv2.matchTemplate(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cp, cv2.TM_CCOEFF_NORMED)
                except:
                    break
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                w, h = cp.shape[::-1]
                bottom_right = (max_loc[0] + w, max_loc[1] + h)
                if max_val > best:
                    cnt = 0
                    best = max_val
                    best_res = max_loc
                    best_right = (max_loc[0] + w, max_loc[1] + h)
                else:
                    cnt += 1
                    if cnt > 2:
                        break
            w, h = temp.shape[::-1]
            top_left = best_res
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(img, top_left, best_right, (0, 255, 0), 3)

            res = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res.thumbnail((600, 600))
            bio = io.BytesIO()
            res.save(bio, format="PNG")
            window['-result-'].update(data=bio.getvalue())
        elif event == '-detect_vj-':
            source_path = values['-load_source-']
            if len(source_path) == 0:
                continue
            img = cv2.imread(source_path, 1)
            detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = detector.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            res = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            res.thumbnail((600, 600))
            bio = io.BytesIO()
            res.save(bio, format="PNG")
            window['-result-'].update(data=bio.getvalue())
        elif event == '-load_source-':
            file_name = values['-load_source-']
            window['-result-'].update(data=io.BytesIO().getvalue())
            try:
                window["-image-"].update(data=load_image(file_name))
            except:
                pass
        elif event == '-load_template-':
            file_name = values['-load_template-']
            window['-result-'].update(data=io.BytesIO().getvalue())
            try:
                window["-template-"].update(data=load_image(file_name))
            except:
                pass
        elif event == sg.WIN_CLOSED:
            break
    window.close()


def load_image(path, size=(600, 600)):
    image = Image.open(path)
    image.thumbnail(size)
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    return bio.getvalue()


if __name__ == '__main__':
    app()
