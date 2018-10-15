dicompath = '/home/seolen/disk/CTav/DICOMDIR'

import pydicom
from pydicom.filereader import read_dicomdir
from os.path import dirname, join

dicom_dir = read_dicomdir(dicompath)
base_dir = dirname(dicompath)

# patients.txt
log_filename = './patients2.txt'
open(log_filename, 'w+').close
log = open(log_filename, 'a+')

for patient_record in dicom_dir.patient_records:
    if (hasattr(patient_record, 'PatientID') and
            hasattr(patient_record, 'PatientName')):
        print("Patient: {}: {}".format(patient_record.PatientID,
                                       patient_record.PatientName))
    studies = patient_record.children
    # got through each serie
    for study in studies:
        print(" " * 4 + "Study {}: {}: {}".format(study.StudyID,
                                                  study.StudyDate,
                                                  study.StudyDescription))
        all_series = study.children
        # go through each serie
        for series in all_series:
            image_count = len(series.children)

            # Write basic series info and image count

            if 'SeriesDescription' not in series:
                series.SeriesDescription = "N/A"
            print(" " * 8 + "Series {}: {}: {} ({} image(s))".format(
                series.SeriesNumber, series.Modality, series.SeriesDescription,
                image_count))

            # Open and read something from each image, for demonstration
            # purposes. For simple quick overview of DICOMDIR, leave the
            # following out
            if image_count > 400:
                log.write("Patient {}   Series {}: {}: {} ({} image(s))\n".format(
                    patient_record.PatientName, series.SeriesNumber, series.Modality, series.SeriesDescription,
                    image_count))
                log.write('~~~\n')
                image_records = series.children
                image_filenames = [join(base_dir, *image_rec.ReferencedFileID)
                                   for image_rec in image_records]

                # reverse image_filenames if the order is inverse
                if image_filenames[0] > image_filenames[1]:
                    image_filenames.reverse()

                print(len(image_filenames))

                for item in image_filenames:
                    log.write(item)
                    log.write('\n')
                log.write('###\n')

