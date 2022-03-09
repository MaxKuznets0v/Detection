import PySimpleGUI as sg
from PIL import Image
import io
from Task1.template_matching import TemplateMatching
from Task1.imageFunctions import display


def app():
    file_input_column = [
        [sg.Text('Выберите изображение')], [sg.FileBrowse('Найти', file_types=(("PNG files", "*.png"),
        ("JPG files", "*.jpg"), ("ALL files", "*")), key='-load_source-', enable_events=True)],
        [sg.Text('Выберите шаблон')], [sg.FileBrowse('Найти', file_types=(("PNG files", "*.png"),
        ("JPG files", "*.jpg"), ("ALL files", "*")), key='-load_template-', enable_events=True)]]

    layout = [
        [sg.Column(file_input_column, justification='center')],
        [sg.Column([[sg.Image(key='-image-'), sg.Image(key='-template-')]], justification='center')],
        [sg.Button('Найти одно', key='-detect-'), sg.Button('Найти все', key='-detect_all-'),
         [sg.Column([[sg.Image(key='-result-')]], justification='center')]]]

    window = sg.Window('Нахождение лиц', layout, resizable=True, finalize=True)
    while True:
        event, values = window.read()

        if event == '-detect-':
            source_path = values['-load_source-']
            template_path = values['-load_template-']
            if len(source_path) == 0 or len(template_path) == 0:
                continue
            tm = TemplateMatching(Image.open(source_path))
            faces = tm.find_one(Image.open(template_path), 50, 60)
            if faces is not None:
                res = display(Image.open(source_path), [faces], False)
            else:
                res = Image.open(source_path)
            bio = io.BytesIO()
            res.thumbnail((600, 600))
            res.save(bio, format="PNG")
            window["-template-"].update(data=io.BytesIO().getvalue())
            window['-image-'].update(data=io.BytesIO().getvalue())
            window['-result-'].update(data=bio.getvalue())
        elif event == '-detect_all-':
            source_path = values['-load_source-']
            template_path = values['-load_template-']
            if len(source_path) == 0 or len(template_path) == 0:
                continue
            tm = TemplateMatching(Image.open(source_path))
            faces = tm.find_all(Image.open(template_path), 50, 60)
            res = display(Image.open(source_path), faces, False)
            bio = io.BytesIO()
            res.thumbnail((600, 600))
            res.save(bio, format="PNG")
            window["-template-"].update(data=io.BytesIO().getvalue())
            window['-image-'].update(data=io.BytesIO().getvalue())
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
