from termcolor import cprint
import os

def remove_corrupt_image(context, full_image_path, file_name):
    # TODO mention in thesis, that this may cause data bias in favour of the 3d or 2d model
    # Remove actual file
    cprint('failed to find a face rect for image: ' + file_name, 'red')
    os.remove(full_image_path)
    cprint('deleted image image from working dir: ' + file_name, 'red')
    # Remove the respective testing entry
    for id, testing_entry in context.open_testing_entry.items():
        if testing_entry.gallery_image_file_name == file_name or testing_entry.input_image_file_name == file_name:
            cprint('removing item from testing_entries with id: ' + str(id), 'red')
            del context.open_testing_entry[id]
            break