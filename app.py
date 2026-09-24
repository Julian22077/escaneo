import cv2
import numpy as np
from rapidocr import RapidOCR
import imutils
import re

from fastapi import FastAPI, UploadFile, File


app = FastAPI(
    title="Lector de Placas",
    description="API para detectar placas vehiculares mediante OpenCV y RapidOCR",
    version="1.0.0"
)


# =========================================================
# CARGAR OCR UNA SOLA VEZ
# =========================================================

ocr = RapidOCR()


# =========================================================
# OBTENER PLACA
# =========================================================

def obteneplaca(location, img, gray):

    mask = np.zeros(gray.shape, np.uint8)

    new_image = cv2.drawContours(
        mask,
        [location],
        0,
        255,
        -1
    )

    new_image = cv2.bitwise_and(
        img,
        img,
        mask=mask
    )

    x, y = np.where(mask == 255)

    if len(x) == 0 or len(y) == 0:
        return None

    x1, y1 = np.min(x), np.min(y)
    x2, y2 = np.max(x), np.max(y)

    cropped_image = gray[
        x1:x2 + 1,
        y1:y2 + 1
    ]


    # =====================================================
    # OCR
    # =====================================================

    result = ocr(cropped_image)


    # =====================================================
    # BUSCAR TEXTO CON FORMATO DE PLACA
    # =====================================================

    if result and result.txts:

        for texto in result.txts:

            # Convertir a mayúsculas
            texto = texto.upper()

            # Eliminar espacios, guiones y caracteres especiales
            texto = re.sub(
                r'[^A-Z0-9]',
                '',
                texto
            )

            # Comprobar formato:
            # 3 letras + 3 números
            if re.fullmatch(
                r'[A-Z]{3}[0-9]{3}',
                texto
            ):
                return texto


    return None


# =========================================================
# DETECTAR PLACA
# =========================================================

def detectar_placa(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    bfilter = cv2.bilateralFilter(
        gray,
        11,
        11,
        17
    )

    edged = cv2.Canny(
        bfilter,
        30,
        200
    )

    keypoints = cv2.findContours(
        edged.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = imutils.grab_contours(
        keypoints
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )[:10]


    for contour in contours:

        approx = cv2.approxPolyDP(
            contour,
            10,
            True
        )

        if len(approx) == 4:

            placa = obteneplaca(
                approx,
                img,
                gray
            )

            if placa:
                return placa


    return None


# =========================================================
# ENDPOINT
# =========================================================

@app.post("/detectar")
async def detectar(
    file: UploadFile = File(...)
):

    contenido = await file.read()

    img = cv2.imdecode(
        np.frombuffer(
            contenido,
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if img is None:

        return {
            "placa": None,
            "error": "No se pudo leer la imagen"
        }


    placa = detectar_placa(img)


    return {
        "placa": placa
    }