from api.background_removal import remove_background


def process_image_for_ocr_and_prediction(image):
    # Remove background
    processed_image = remove_background(image)

    # Call your OCR and shelf life prediction functions here
    # For example:
    # ocr_result = ocr_function(processed_image)
    # shelf_life_result = shelf_life_prediction_function(processed_image)

    # Return or combine results as needed
    return processed_image  # Modify this to return desired results
