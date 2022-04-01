import PySimpleGUI as sg
from Task2.fares import FaReS
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Task2.methods import *
from Task2.analyze import assign_param, validate
import matplotlib.pyplot as plt


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


fig_agg = None


def app(predictor):
    global fig_agg
    cur_size = 0
    layout = [[sg.Canvas(key='-canvas-')],
              [sg.Button('Далее', key='-load_image-'), sg.Button('Подбор параметров', key='-parameters-'),
               sg.Button('Настройка эталонов', key='-train_size-'),
               sg.Button('Точность от числа тестовых изображений', key='-full_test-')],
               [sg.FileBrowse('Найти', key='-pic-', enable_events=True, initial_folder="ORL")]]

    window = sg.Window('FaReS', layout, resizable=True, finalize=True, size=(1000, 500), element_justification="center")
    while True:
        event, values = window.read()
        if event == '-load_image-':
            file_name = predictor.pick_next()[2]
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            *_, fig = predictor.predict(img, True)
            if fig_agg is not None:
                fig_agg.get_tk_widget().forget()
            fig_agg = draw_figure(window['-canvas-'].TKCanvas, fig)
        elif event == '-parameters-':
            cur_size += 1
            if cur_size > 10:
                cur_size = 9
            methods = [Histogram, DFT, DCT, Scale, Gradient]
            plt.clf()
            collection, test = build_collection(build_targets('orig'), cur_size)
            for method in methods:
                assign_param(collection, test, method)
            fig = plt.gcf()
            plt.title(f'Train size = {cur_size}')
            plt.legend()
            if fig_agg is not None:
                fig_agg.get_tk_widget().forget()
            fig_agg = draw_figure(window['-canvas-'].TKCanvas, fig)
        elif event == '-train_size-':
            plt.clf()
            plt.plot(range(1, 10), validate(FaReS, 10))
            fig = plt.gcf()
            if fig_agg is not None:
                fig_agg.get_tk_widget().forget()
            fig_agg = draw_figure(window['-canvas-'].TKCanvas, fig)
        elif event == '-full_test-':
            plt.clf()
            plt.plot(predictor.test_graph())
            fig = plt.gcf()
            if fig_agg is not None:
                fig_agg.get_tk_widget().forget()
            fig_agg = draw_figure(window['-canvas-'].TKCanvas, fig)
        elif event == '-pic-':
            file_name = values['-pic-']
            if file_name == '':
                continue
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            *_, fig = predictor.predict(img, True)
            if fig_agg is not None:
                fig_agg.get_tk_widget().forget()
            fig_agg = draw_figure(window['-canvas-'].TKCanvas, fig)
        elif event == sg.WIN_CLOSED:
            break


if __name__ == '__main__':
    predictor = FaReS('orig', 41, 5)
    app(predictor)
