#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:46:27 2017

@author: rsburugula
"""
import sys
import pyedflib
import numpy as np
f = pyedflib.EdfReader(sys.argv[1])
n = f.signals_in_file
print ("number of signals in file = ", n)
signal_labels = f.getSignalLabels()
print ("signal labels = ", signal_labels)

print("\nlibrary version: %s" % pyedflib.version.version)

print("\ngeneral header:\n")

# print("filetype: %i\n"%hdr.filetype);
print("edfsignals: %i" % f.signals_in_file)
print("file duration: %i seconds" % f.file_duration)
print("startdate: %i-%i-%i" % (f.getStartdatetime().day,f.getStartdatetime().month,f.getStartdatetime().year))
print("starttime: %i:%02i:%02i" % (f.getStartdatetime().hour,f.getStartdatetime().minute,f.getStartdatetime().second))
# print("patient: %s" % f.getP);
# print("recording: %s" % f.getPatientAdditional())
print("patientcode: %s" % f.getPatientCode())
print("gender: %s" % f.getGender())
print("birthdate: %s" % f.getBirthdate())
print("patient_name: %s" % f.getPatientName())
print("patient_additional: %s" % f.getPatientAdditional())
print("admincode: %s" % f.getAdmincode())
print("technician: %s" % f.getTechnician())
print("equipment: %s" % f.getEquipment())
print("recording_additional: %s" % f.getRecordingAdditional())
print("datarecord duration: %f seconds" % f.getFileDuration())
print("number of datarecords in the file: %i" % f.datarecords_in_file)
print("number of annotations in the file: %i" % f.annotations_in_file)

channel = 3
print("\nsignal parameters for the %d.channel:\n\n" % channel)

print("label: %s" % f.getLabel(channel))
print("samples in file: %i" % f.getNSamples()[channel])
# print("samples in datarecord: %i" % f.get
print("physical maximum: %f" % f.getPhysicalMaximum(channel))
print("physical minimum: %f" % f.getPhysicalMinimum(channel))
print("digital maximum: %i" % f.getDigitalMaximum(channel))
print("digital minimum: %i" % f.getDigitalMinimum(channel))
print("physical dimension: %s" % f.getPhysicalDimension(channel))
print("prefilter: %s" % f.getPrefilter(channel))
print("transducer: %s" % f.getTransducer(channel))
print("samplefrequency: %f" % f.getSampleFrequency(channel))

#buf = f.readSignal(channel)
#n = 200
#print("\nread %i samples\n" % n)
#result = ""
#for i in np.arange(n):
#    result += ("%.1f, " % buf[i])
#print("result = ", result)

sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)

print (sigbufs.shape)