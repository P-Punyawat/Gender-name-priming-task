#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on ??????? 29, 2025, at 13:24
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_3
import serial
#port = serial.Serial('COM4',115200)
#
# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'Priming_gender'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\NARONG\\Downloads\\Gender-name-priming-task-main\\Priming_gender_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('instruction_continue') is None:
        # initialise instruction_continue
        instruction_continue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction_continue',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('instruction_continue_2') is None:
        # initialise instruction_continue_2
        instruction_continue_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction_continue_2',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "training_Instruction" ---
    instruction_slide = visual.ImageStim(
        win=win,
        name='instruction_slide', units='norm', 
        image='Instruction.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=True, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    instruction_continue = keyboard.Keyboard(deviceName='instruction_continue')
    
    # --- Initialize components for Routine "training_block" ---
    MAN = visual.TextStim(win=win, name='MAN',
        text='MAN',
        font='Arial',
        units='norm', pos=(0.8, 0.8), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    WOMAN = visual.TextStim(win=win, name='WOMAN',
        text='WOMAN',
        font='Arial',
        units='norm', pos=(-0.8, 0.8), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    fixation = visual.ShapeStim(
        win=win, name='fixation',
        size=(0.1, 0.1), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=True, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor=[-1.0000, -1.0000, 1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    pic = visual.ImageStim(
        win=win,
        name='pic', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.0, 0.0), draggable=False, size=(1.2, 1.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    error_feedback = visual.ShapeStim(
        win=win, name='error_feedback', vertices='cross',units='norm', 
        size=(0.5, 0.5),
        ori=45.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='red', fillColor='red',
        opacity=None, depth=-5.0, interpolate=True)
    # Run 'Begin Experiment' code from code
    
    
    
    # --- Initialize components for Routine "priming_instruction" ---
    instruction_slide_2 = visual.ImageStim(
        win=win,
        name='instruction_slide_2', units='norm', 
        image='Instruction.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=True, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    instruction_continue_2 = keyboard.Keyboard(deviceName='instruction_continue_2')
    
    # --- Initialize components for Routine "priming_block" ---
    MAN_3 = visual.TextStim(win=win, name='MAN_3',
        text='MAN',
        font='Arial',
        units='norm', pos=(0.8, 0.8), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    MAN_3_desc = visual.TextStim(win=win, name='MAN_3_desc',
        text='Press the k key',
        font='Arial',
        units='norm', pos=(0.8, 0.7), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    WOMAN_3 = visual.TextStim(win=win, name='WOMAN_3',
        text='WOMAN',
        font='Arial',
        units='norm', pos=(-0.8, 0.8), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    WOMAN_3_desc = visual.TextStim(win=win, name='WOMAN_3_desc',
        text='Press the j key',
        font='Arial',
        units='norm', pos=(-0.8, 0.7), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    name_2 = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Angsana New',
         ori=0.0, pos=(0, 0), draggable=False, units='norm',     letterHeight=0.5,
         size=(1, 1), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='name_2',
         depth=-5, autoLog=True,
    )
    pic_2 = visual.ImageStim(
        win=win,
        name='pic_2', units='norm', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0.0, 0.0), draggable=False, size=(1.2, 1.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "Ending" ---
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "training_Instruction" ---
    # create an object to store info about Routine training_Instruction
    training_Instruction = data.Routine(
        name='training_Instruction',
        components=[instruction_slide, instruction_continue],
    )
    training_Instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction_continue
    instruction_continue.keys = []
    instruction_continue.rt = []
    _instruction_continue_allKeys = []
    # store start times for training_Instruction
    training_Instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    training_Instruction.tStart = globalClock.getTime(format='float')
    training_Instruction.status = STARTED
    thisExp.addData('training_Instruction.started', training_Instruction.tStart)
    training_Instruction.maxDuration = None
    # keep track of which components have finished
    training_InstructionComponents = training_Instruction.components
    for thisComponent in training_Instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "training_Instruction" ---
    training_Instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_slide* updates
        
        # if instruction_slide is starting this frame...
        if instruction_slide.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_slide.frameNStart = frameN  # exact frame index
            instruction_slide.tStart = t  # local t and not account for scr refresh
            instruction_slide.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_slide, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_slide.started')
            # update status
            instruction_slide.status = STARTED
            instruction_slide.setAutoDraw(True)
        
        # if instruction_slide is active this frame...
        if instruction_slide.status == STARTED:
            # update params
            pass
        
        # *instruction_continue* updates
        waitOnFlip = False
        
        # if instruction_continue is starting this frame...
        if instruction_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_continue.frameNStart = frameN  # exact frame index
            instruction_continue.tStart = t  # local t and not account for scr refresh
            instruction_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_continue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_continue.started')
            # update status
            instruction_continue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction_continue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction_continue.status == STARTED and not waitOnFlip:
            theseKeys = instruction_continue.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction_continue_allKeys.extend(theseKeys)
            if len(_instruction_continue_allKeys):
                instruction_continue.keys = _instruction_continue_allKeys[-1].name  # just the last key pressed
                instruction_continue.rt = _instruction_continue_allKeys[-1].rt
                instruction_continue.duration = _instruction_continue_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=training_Instruction,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            training_Instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in training_Instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "training_Instruction" ---
    for thisComponent in training_Instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for training_Instruction
    training_Instruction.tStop = globalClock.getTime(format='float')
    training_Instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('training_Instruction.stopped', training_Instruction.tStop)
    # check responses
    if instruction_continue.keys in ['', [], None]:  # No response was made
        instruction_continue.keys = None
    thisExp.addData('instruction_continue.keys',instruction_continue.keys)
    if instruction_continue.keys != None:  # we had a response
        thisExp.addData('instruction_continue.rt', instruction_continue.rt)
        thisExp.addData('instruction_continue.duration', instruction_continue.duration)
    thisExp.nextEntry()
    # the Routine "training_Instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    training = data.TrialHandler2(
        name='training',
        nReps=20.0, 
        method='fullRandom', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'Training loop.csv', 
        selection='0:2'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(training)  # add the loop to the experiment
    thisTraining = training.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTraining.rgb)
    if thisTraining != None:
        for paramName in thisTraining:
            globals()[paramName] = thisTraining[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTraining in training:
        training.status = STARTED
        if hasattr(thisTraining, 'status'):
            thisTraining.status = STARTED
        currentLoop = training
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTraining.rgb)
        if thisTraining != None:
            for paramName in thisTraining:
                globals()[paramName] = thisTraining[paramName]
        
        # --- Prepare to start Routine "training_block" ---
        # create an object to store info about Routine training_block
        training_block = data.Routine(
            name='training_block',
            components=[MAN, WOMAN, fixation, pic, key_resp, error_feedback],
        )
        training_block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        pic.setImage(Image)
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for training_block
        training_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        training_block.tStart = globalClock.getTime(format='float')
        training_block.status = STARTED
        thisExp.addData('training_block.started', training_block.tStart)
        training_block.maxDuration = None
        # keep track of which components have finished
        training_blockComponents = training_block.components
        for thisComponent in training_block.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "training_block" ---
        training_block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTraining, 'status') and thisTraining.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *MAN* updates
            
            # if MAN is starting this frame...
            if MAN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                MAN.frameNStart = frameN  # exact frame index
                MAN.tStart = t  # local t and not account for scr refresh
                MAN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(MAN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'MAN.started')
                # update status
                MAN.status = STARTED
                MAN.setAutoDraw(True)
            
            # if MAN is active this frame...
            if MAN.status == STARTED:
                # update params
                pass
            
            # *WOMAN* updates
            
            # if WOMAN is starting this frame...
            if WOMAN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                WOMAN.frameNStart = frameN  # exact frame index
                WOMAN.tStart = t  # local t and not account for scr refresh
                WOMAN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(WOMAN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'WOMAN.started')
                # update status
                WOMAN.status = STARTED
                WOMAN.setAutoDraw(True)
            
            # if WOMAN is active this frame...
            if WOMAN.status == STARTED:
                # update params
                pass
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *pic* updates
            
            # if pic is starting this frame...
            if pic.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                pic.frameNStart = frameN  # exact frame index
                pic.tStart = t  # local t and not account for scr refresh
                pic.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pic, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pic.started')
                # update status
                pic.status = STARTED
                pic.setAutoDraw(True)
            
            # if pic is active this frame...
            if pic.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['j','k'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *error_feedback* updates
            
            # if error_feedback is active this frame...
            if error_feedback.status == STARTED:
                # update params
                pass
            # Run 'Each Frame' code from code
            # Only show feedback if a key has been pressed
            if key_resp.keys:  # keyResp is your Keyboard component
                # Determine the correct key based on the image
                if "men.png" in pic.image:  # imageStim is your Image component
                    correctKey = 'k'
                elif "women.png" in pic.image:
                    correctKey = 'j'
                
                # Show cross if wrong
                if key_resp.keys[0] != correctKey:
                    error_feedback.setAutoDraw(True)
                else:
                    error_feedback.setAutoDraw(False)
                    
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=training_block,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                training_block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in training_block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "training_block" ---
        for thisComponent in training_block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for training_block
        training_block.tStop = globalClock.getTime(format='float')
        training_block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('training_block.stopped', training_block.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        training.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            training.addData('key_resp.rt', key_resp.rt)
            training.addData('key_resp.duration', key_resp.duration)
        # the Routine "training_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTraining as finished
        if hasattr(thisTraining, 'status'):
            thisTraining.status = FINISHED
        # if awaiting a pause, pause now
        if training.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            training.status = STARTED
        thisExp.nextEntry()
        
    # completed 20.0 repeats of 'training'
    training.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "priming_instruction" ---
    # create an object to store info about Routine priming_instruction
    priming_instruction = data.Routine(
        name='priming_instruction',
        components=[instruction_slide_2, instruction_continue_2],
    )
    priming_instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction_continue_2
    instruction_continue_2.keys = []
    instruction_continue_2.rt = []
    _instruction_continue_2_allKeys = []
    # store start times for priming_instruction
    priming_instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    priming_instruction.tStart = globalClock.getTime(format='float')
    priming_instruction.status = STARTED
    thisExp.addData('priming_instruction.started', priming_instruction.tStart)
    priming_instruction.maxDuration = None
    # keep track of which components have finished
    priming_instructionComponents = priming_instruction.components
    for thisComponent in priming_instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "priming_instruction" ---
    priming_instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instruction_slide_2* updates
        
        # if instruction_slide_2 is starting this frame...
        if instruction_slide_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_slide_2.frameNStart = frameN  # exact frame index
            instruction_slide_2.tStart = t  # local t and not account for scr refresh
            instruction_slide_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_slide_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_slide_2.started')
            # update status
            instruction_slide_2.status = STARTED
            instruction_slide_2.setAutoDraw(True)
        
        # if instruction_slide_2 is active this frame...
        if instruction_slide_2.status == STARTED:
            # update params
            pass
        
        # *instruction_continue_2* updates
        waitOnFlip = False
        
        # if instruction_continue_2 is starting this frame...
        if instruction_continue_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction_continue_2.frameNStart = frameN  # exact frame index
            instruction_continue_2.tStart = t  # local t and not account for scr refresh
            instruction_continue_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction_continue_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction_continue_2.started')
            # update status
            instruction_continue_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction_continue_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction_continue_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction_continue_2.status == STARTED and not waitOnFlip:
            theseKeys = instruction_continue_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction_continue_2_allKeys.extend(theseKeys)
            if len(_instruction_continue_2_allKeys):
                instruction_continue_2.keys = _instruction_continue_2_allKeys[-1].name  # just the last key pressed
                instruction_continue_2.rt = _instruction_continue_2_allKeys[-1].rt
                instruction_continue_2.duration = _instruction_continue_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=priming_instruction,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            priming_instruction.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in priming_instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "priming_instruction" ---
    for thisComponent in priming_instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for priming_instruction
    priming_instruction.tStop = globalClock.getTime(format='float')
    priming_instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('priming_instruction.stopped', priming_instruction.tStop)
    # check responses
    if instruction_continue_2.keys in ['', [], None]:  # No response was made
        instruction_continue_2.keys = None
    thisExp.addData('instruction_continue_2.keys',instruction_continue_2.keys)
    if instruction_continue_2.keys != None:  # we had a response
        thisExp.addData('instruction_continue_2.rt', instruction_continue_2.rt)
        thisExp.addData('instruction_continue_2.duration', instruction_continue_2.duration)
    thisExp.nextEntry()
    # the Routine "priming_instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    priming = data.TrialHandler2(
        name='priming',
        nReps=5.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Training loop.csv'), 
        seed=None, 
    )
    thisExp.addLoop(priming)  # add the loop to the experiment
    thisPriming = priming.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPriming.rgb)
    if thisPriming != None:
        for paramName in thisPriming:
            globals()[paramName] = thisPriming[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPriming in priming:
        priming.status = STARTED
        if hasattr(thisPriming, 'status'):
            thisPriming.status = STARTED
        currentLoop = priming
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPriming.rgb)
        if thisPriming != None:
            for paramName in thisPriming:
                globals()[paramName] = thisPriming[paramName]
        
        # --- Prepare to start Routine "priming_block" ---
        # create an object to store info about Routine priming_block
        priming_block = data.Routine(
            name='priming_block',
            components=[MAN_3, MAN_3_desc, WOMAN_3, WOMAN_3_desc, name_2, pic_2, key_resp_3],
        )
        priming_block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_4
        stimulus_name_started = False
        stimulus_name_ended = False
        
        stimulus_pic_started = False
        stimulus_pic_ended = False
        
        
        #win.callOnFlip(port.write, Trig.to_bytes(1,'little'))
        #win.callOnFlip(port.write, int(Trig).to_bytes(1, 'little'))
        win.callOnFlip(print, "Trial Trigger sent:", int(Trig))
        #block_start_time = globalClock.getTime()
        #if globalClock.getTime() - block_start_time >= 0.015:
        #       win.callOnFlip(port.write, b'\x00')
        #
        name_2.reset()
        name_2.setText(NamePriming
        
        
        )
        pic_2.setImage(ImagePriming)
        # create starting attributes for key_resp_3
        key_resp_3.keys = []
        key_resp_3.rt = []
        _key_resp_3_allKeys = []
        # store start times for priming_block
        priming_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        priming_block.tStart = globalClock.getTime(format='float')
        priming_block.status = STARTED
        thisExp.addData('priming_block.started', priming_block.tStart)
        priming_block.maxDuration = None
        # keep track of which components have finished
        priming_blockComponents = priming_block.components
        for thisComponent in priming_block.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "priming_block" ---
        priming_block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisPriming, 'status') and thisPriming.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code_4
            # send trigger to port, converting to bytes
            # acceptable trigger values are 1-255 (integer values)
            if name_2.status == STARTED and not stimulus_name_started:
            #    win.callOnFlip(port.write, str.encode(trig))
                win.callOnFlip(print, "Name Trigger sent:", int(NameTrig))
            #    win.callOnFlip(port.write, int(NameTrig).to_bytes(1,'little'))
                stimulus_name_start_time = globalClock.getTime()
                stimulus_name_started  = True
            
            if pic_2.status == STARTED and not stimulus_pic_started:
            #    win.callOnFlip(port.write, str.encode(trig))
                win.callOnFlip(print, "Pic Trigger sent:", int(ImageTrig))
            #    win.callOnFlip(port.write, int(ImageTrig).to_bytes(1,'little'))
                stimulus_pic_start_time = globalClock.getTime()
                stimulus_pic_started  = True
                
            ## #set port back to zero after 15 ms
            #if stimulus_name_started and not stimulus_name_ended:
            #        if globalClock.getTime() - stimulus_name_start_time >= 0.015:
            #            win.callOnFlip(port.write, b'\x00')
            #            stimulus_name_ended = True
            #if stimulus_pic_started and not stimulus_pic_ended:
            #        if globalClock.getTime() - stimulus_pic_start_time >= 0.015:
            #            win.callOnFlip(port.write, b'\x00')
            #            stimulus_pic_ended = True
            
            # *MAN_3* updates
            
            # if MAN_3 is starting this frame...
            if MAN_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                MAN_3.frameNStart = frameN  # exact frame index
                MAN_3.tStart = t  # local t and not account for scr refresh
                MAN_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(MAN_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'MAN_3.started')
                # update status
                MAN_3.status = STARTED
                MAN_3.setAutoDraw(True)
            
            # if MAN_3 is active this frame...
            if MAN_3.status == STARTED:
                # update params
                pass
            
            # *MAN_3_desc* updates
            
            # if MAN_3_desc is starting this frame...
            if MAN_3_desc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                MAN_3_desc.frameNStart = frameN  # exact frame index
                MAN_3_desc.tStart = t  # local t and not account for scr refresh
                MAN_3_desc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(MAN_3_desc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'MAN_3_desc.started')
                # update status
                MAN_3_desc.status = STARTED
                MAN_3_desc.setAutoDraw(True)
            
            # if MAN_3_desc is active this frame...
            if MAN_3_desc.status == STARTED:
                # update params
                pass
            
            # *WOMAN_3* updates
            
            # if WOMAN_3 is starting this frame...
            if WOMAN_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                WOMAN_3.frameNStart = frameN  # exact frame index
                WOMAN_3.tStart = t  # local t and not account for scr refresh
                WOMAN_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(WOMAN_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'WOMAN_3.started')
                # update status
                WOMAN_3.status = STARTED
                WOMAN_3.setAutoDraw(True)
            
            # if WOMAN_3 is active this frame...
            if WOMAN_3.status == STARTED:
                # update params
                pass
            
            # *WOMAN_3_desc* updates
            
            # if WOMAN_3_desc is starting this frame...
            if WOMAN_3_desc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                WOMAN_3_desc.frameNStart = frameN  # exact frame index
                WOMAN_3_desc.tStart = t  # local t and not account for scr refresh
                WOMAN_3_desc.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(WOMAN_3_desc, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'WOMAN_3_desc.started')
                # update status
                WOMAN_3_desc.status = STARTED
                WOMAN_3_desc.setAutoDraw(True)
            
            # if WOMAN_3_desc is active this frame...
            if WOMAN_3_desc.status == STARTED:
                # update params
                pass
            
            # *name_2* updates
            
            # if name_2 is starting this frame...
            if name_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                name_2.frameNStart = frameN  # exact frame index
                name_2.tStart = t  # local t and not account for scr refresh
                name_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(name_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'name_2.started')
                # update status
                name_2.status = STARTED
                name_2.setAutoDraw(True)
            
            # if name_2 is active this frame...
            if name_2.status == STARTED:
                # update params
                pass
            
            # if name_2 is stopping this frame...
            if name_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > name_2.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    name_2.tStop = t  # not accounting for scr refresh
                    name_2.tStopRefresh = tThisFlipGlobal  # on global time
                    name_2.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'name_2.stopped')
                    # update status
                    name_2.status = FINISHED
                    name_2.setAutoDraw(False)
            
            # *pic_2* updates
            
            # if pic_2 is starting this frame...
            if pic_2.status == NOT_STARTED and tThisFlip >= 0.3-frameTolerance:
                # keep track of start time/frame for later
                pic_2.frameNStart = frameN  # exact frame index
                pic_2.tStart = t  # local t and not account for scr refresh
                pic_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(pic_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'pic_2.started')
                # update status
                pic_2.status = STARTED
                pic_2.setAutoDraw(True)
            
            # if pic_2 is active this frame...
            if pic_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_3* updates
            waitOnFlip = False
            
            # if key_resp_3 is starting this frame...
            if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_3.frameNStart = frameN  # exact frame index
                key_resp_3.tStart = t  # local t and not account for scr refresh
                key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_3.started')
                # update status
                key_resp_3.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_3.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_3.getKeys(keyList=['j','k'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_3_allKeys.extend(theseKeys)
                if len(_key_resp_3_allKeys):
                    key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                    key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                    key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=priming_block,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                priming_block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in priming_block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "priming_block" ---
        for thisComponent in priming_block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for priming_block
        priming_block.tStop = globalClock.getTime(format='float')
        priming_block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('priming_block.stopped', priming_block.tStop)
        # check responses
        if key_resp_3.keys in ['', [], None]:  # No response was made
            key_resp_3.keys = None
        priming.addData('key_resp_3.keys',key_resp_3.keys)
        if key_resp_3.keys != None:  # we had a response
            priming.addData('key_resp_3.rt', key_resp_3.rt)
            priming.addData('key_resp_3.duration', key_resp_3.duration)
        # the Routine "priming_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisPriming as finished
        if hasattr(thisPriming, 'status'):
            thisPriming.status = FINISHED
        # if awaiting a pause, pause now
        if priming.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            priming.status = STARTED
        thisExp.nextEntry()
        
    # completed 5.0 repeats of 'priming'
    priming.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Ending" ---
    # create an object to store info about Routine Ending
    Ending = data.Routine(
        name='Ending',
        components=[],
    )
    Ending.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_5
    #port.close()
    # store start times for Ending
    Ending.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Ending.tStart = globalClock.getTime(format='float')
    Ending.status = STARTED
    thisExp.addData('Ending.started', Ending.tStart)
    Ending.maxDuration = 1
    # keep track of which components have finished
    EndingComponents = Ending.components
    for thisComponent in Ending.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Ending" ---
    Ending.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # is it time to end the Routine? (based on local clock)
        if tThisFlip > Ending.maxDuration-frameTolerance:
            Ending.maxDurationReached = True
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=Ending,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Ending.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Ending.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Ending" ---
    for thisComponent in Ending.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Ending
    Ending.tStop = globalClock.getTime(format='float')
    Ending.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Ending.stopped', Ending.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Ending.maxDurationReached:
        routineTimer.addTime(-Ending.maxDuration)
    elif Ending.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
