import subprocess
import os
import shutil
import tempfile

# Путь к директории с maltparser (где лежат jar и .mco)
MALT_DIR = "/Users/mac/Desktop/vscode/gromov/maltparser"
JAR_NAME = "maltparser-1.9.2.jar"
JAR_PATH = os.path.join(MALT_DIR, JAR_NAME)

# Имя модели, как ты запускала в терминале (без .mco)
MODEL_NAME = "engmalt.linear-1.7"

def run_malt(input_path, output_path=None):
    """
    Запускает MaltParser на указанном входном .conll-файле.
    Логика:
      - если input_path уже в MALT_DIR и называется 'input.conll', запускаем прямо;
      - иначе копируем input_path -> MALT_DIR/input.conll, запускаем в cwd=MALT_DIR,
        после чего читаем output_path и возвращаем строку.
    Возвращает текст результата (CoNLL) или выбрасывает исключение subprocess.CalledProcessError.
    """
    if not os.path.exists(JAR_PATH):
        raise FileNotFoundError(f"Cannot find jar at {JAR_PATH}. Check MALT_DIR.")

    abs_input = os.path.abspath(input_path)

    # Имя входного файла внутри MALT_DIR
    malt_input_name = "input.conll"
    malt_input = os.path.join(MALT_DIR, malt_input_name)

    # Если input файл уже в MALT_DIR с нужным именем, не копируем
    if not (os.path.abspath(os.path.dirname(abs_input)) == os.path.abspath(MALT_DIR) and os.path.basename(abs_input) == malt_input_name):
        shutil.copy2(abs_input, malt_input)
    else:
        malt_input = abs_input

    # Если output_path не задан, создаём рядом со скриптом с именем output.conll
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "output.conll")
    else:
        output_path = os.path.abspath(output_path)

    # Удаляем старый output, если он есть, чтобы не путаться
    if os.path.exists(output_path):
        os.remove(output_path)

    cmd = [
        "java", "-Xmx1024m", "-jar", JAR_PATH,
        "-c", MODEL_NAME,
        "-i", malt_input,
        "-o", output_path,
        "-m", "parse"
    ]

    # Запускаем в рабочей директории MALT_DIR (как ты делала вручную в терминале)
    subprocess.run(cmd, check=True, cwd=MALT_DIR)

    # Читаем результат
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"MaltParser finished but did not produce {output_path}")

    with open(output_path, "r", encoding="utf-8") as f:
        parsed = f.read()

    # Очистим временную копию входа, если копировали
    if malt_input != abs_input and os.path.exists(malt_input):
        try:
            os.remove(malt_input)
        except Exception:
            pass

    # Оставляем output в папке скрипта (рядом с ним)

    return parsed

if __name__ == "__main__":
    # Пример: запустится на ./input.conll (в текущей папке проекта)
    # Если хочешь запускать другой файл, передай путь в run_malt(...)
    input_file = "input.conll"
    text = run_malt(input_file)
    print(text)