import PySimpleGUI as sg
from Task3.detect import StyleDetector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import matplotlib.pyplot as plt


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


fig_agg = None


def app(predictor):
    global fig_agg
    layout = [[sg.Canvas(key='-canvas-')],
              [sg.FileBrowse('Выбрать', key='-load_image-', enable_events=True, initial_folder="database")],
              [sg.Text('Найденный стиль: ', key='-res-')]]

    window = sg.Window('Detector', layout, resizable=True, finalize=True, size=(1000, 500), element_justification="center")
    while True:
        event, values = window.read()
        if event == '-load_image-':
            file_name = values['-load_image-']
            if file_name == '':
                continue
            img = cv2.imread(file_name)
            features = dict()
            features['Original'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            features.update(predictor.get_features(img))
            fig = plt.figure(figsize=(12, 3))
            columns = 4
            rows = 2
            guess, paths = predictor.predict(img)

            for i, name in enumerate(features):
                if i > 3:
                    sub = fig.add_subplot(rows, columns, i + 2)
                    sub.title.set_text(name)
                    plt.imshow(features[name], aspect="auto")
                else:
                    sub = fig.add_subplot(rows, columns, i + 1)
                    sub.title.set_text(name)
                    if name == 'ColorHistogram':
                        colors = ('b', 'g', 'r')
                        for cl in range(3):
                            plt.plot(features[name][cl], color=colors[cl])
                    elif name == 'SIFT':
                        img_rs = cv2.resize(img, (200, 200))
                        res = cv2.drawKeypoints(cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB), features[name], None, (255, 0, 255))
                        plt.imshow(res, aspect="auto")
                    else:
                        plt.imshow(features[name], aspect="auto")
                plt.xticks([])
                plt.yticks([])
            for i in range(4, 7):
                sub = fig.add_subplot(rows, columns, i+2)
                sub.title.set_text(paths[i-4][0])
                plt.imshow(paths[i-4][1], aspect="auto")
                plt.xticks([])
                plt.yticks([])
            if fig_agg is not None:
                fig_agg.get_tk_widget().forget()
            fig_agg = draw_figure(window['-canvas-'].TKCanvas, fig)
            window['-res-'].update(f"Найденный стиль: {guess}")
        elif event == sg.WIN_CLOSED:
            break


if __name__ == '__main__':
    predictor = StyleDetector('database', 8)
    app(predictor)
