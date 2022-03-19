import PySimpleGUI as sg
from Task2.fares import FaReS
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


fig_agg = None


def app(predictor):
    layout = [[sg.Canvas(key='-canvas-')],
              [sg.FileBrowse('Найти', key='-load_image-', enable_events=True)]]

    window = sg.Window('FaReS', layout, resizable=True, finalize=True, size=(1000, 500), element_justification="center")
    while True:
        event, values = window.read()
        if event == '-load_image-':
            file_name = values['-load_image-']
            if file_name == '':
                continue
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            *_, fig = predictor.predict(img, True)
            global fig_agg
            if fig_agg is not None:
                fig_agg.get_tk_widget().forget()
            fig_agg = draw_figure(window['-canvas-'].TKCanvas, fig)
        elif event == sg.WIN_CLOSED:
            break


if __name__ == '__main__':
    predictor = FaReS('ORL', 41)
    app(predictor)
