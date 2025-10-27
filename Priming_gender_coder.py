# --- IMPORTS ---
from psychopy import visual, core, event, gui, data
from psychopy.visual.textbox2 import TextBox2
from pylsl import StreamInfo, StreamOutlet
import os, random, pandas as pd
import serial 


# --- PATHS ---
_thisDir = os.path.dirname(os.path.abspath(__file__))
face_path = os.path.join(_thisDir, "facebank_clean")
csv_file = os.path.join(_thisDir, "permutations.csv")  # contains Name, Face, Gender
training_faces = [f"face{i}_m.png" for i in range(1,11)] + [f"face{i}_w.png" for i in range(1,11)]
random.shuffle(training_faces)

# ============ SETUP PARTICIPANT INFO ============
expName = "Priming_Gender"
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(expInfo, title=expName)
if not dlg.OK:
    core.quit()

# Setup data handler
filename = f"data/{expInfo['participant']}_{expInfo['session']}_{data.getDateStr()}"
thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=True, saveWideText=True,
    dataFileName=filename
)

# --- LOAD CSV & TRIAL LIST ---
trials_df = pd.read_csv(csv_file)
remaining_trials = list(trials_df.itertuples(index=False, name=None))
random.shuffle(remaining_trials)

# --- TRIGGER CODES ---
face_ids = {f: i+1 for i, f in enumerate(sorted(trials_df['Face'].unique()))}
name_ids = {n: i+21 for i, n in enumerate(trials_df['Name'].unique())}
combo_ids = {"mm":71, "mf":72, "ff":73, "fm":74}
end_trial_code = 90
timeout_code = 91
j_key_id = 81
k_key_id = 82

# --- WINDOW & STIMULI ---
win = visual.Window([1920,1080],screen=1, units="norm", fullscr=False, color='white')
Priming_Name = TextBox2(win, text='', pos=(0,0), font='Noto Sans Thai', color='black',anchor='center',alignment='center',letterHeight=0.3, units='norm')
Priming_Face = visual.ImageStim(win, image=None, pos=(0,0), size=(1,1))
training_instruction_slide = visual.ImageStim(
        win=win,
        name='training_instruction_slide', units='norm', 
        image='Instruction/training_instruction_wm.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=True, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
priming_instruction_slide = visual.ImageStim(
        win=win,
        name='priming_instruction_slide', units='norm', 
        image='Instruction/priming_instruction_wm.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=True, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
MAN_3 = visual.TextStim(win=win, name='MAN_3',
        text='MAN',
        font='Arial',
        units='norm', pos=(0.8, 0.8), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
MAN_3_desc = visual.TextStim(win=win, name='MAN_3_desc',
        text='Press the k key',
        font='Arial',
        units='norm', pos=(0.8, 0.7), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
WOMAN_3 = visual.TextStim(win=win, name='WOMAN_3',
        text='WOMAN',
        font='Arial',
        units='norm', pos=(-0.8, 0.8), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
WOMAN_3_desc = visual.TextStim(win=win, name='WOMAN_3_desc',
        text='Press the j key',
        font='Arial',
        units='norm', pos=(-0.8, 0.7), draggable=False, height=0.06, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0); 
        
# --- SERIAL SETUP ---   
port = serial.Serial('COM3',115200)

# --- LSL SETUP ---
info = StreamInfo(name='PsychoPy_LSL', type='Markers', channel_count=1,
                  nominal_srate=0, channel_format='int32', source_id='psy_marker')
outlet = StreamOutlet(info)

def send_trigger(code):
    """Send trigger to serial port and LSL (if available)."""
    # --- Serial trigger ---
    if port is not None:
        try:
            port.write(bytes([code]))  # send single-byte trigger
            core.wait(0.005)          # short pulse duration (~5 ms)
            port.write(bytes([0]))     # reset line to 0
        except Exception as e:
            print(f"Serial trigger failed: {e}")

    # --- LSL trigger (if you're also using outlet) ---
    try:
        outlet.push_sample([code])
    except Exception as e:
        print(f"LSL trigger failed: {e}")

    print(f"Trigger sent: {code}")

for face_file in training_faces:
    Priming_Face.setImage(os.path.join(face_path, face_file))
    MAN_3.draw()
    MAN_3_desc.draw()
    WOMAN_3.draw()
    WOMAN_3_desc.draw()
    Priming_Face.draw()
    win.flip()
    keys = event.getKeys(keyList=['escape'])
    if 'escape' in keys:
            thisExp.saveAsWideText(filename + '.csv')
            thisExp.saveAsPickle(filename)
            thisExp.abort()
            core.quit()

    keys = event.waitKeys(keyList=['j','k'])
    print(f"Training face {face_file}, key pressed: {keys[0]}")
    
    core.wait(0.5)  # inter-trial interval
    
priming_instruction_slide.draw()
win.flip()

# Wait for SPACE key to continue
print("Showing instructions... (Press SPACE to start)")
while True:
    keys = event.getKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        core.quit()
    elif 'space' in keys:
        print("SPACE pressed — starting experiment.")
        break
    core.wait(0.01)

# Clear instruction screen
win.flip()
core.wait(0.2)
# ============ TRIAL LOOP ============
trialClock = core.Clock()
globalClock = core.Clock()

for current_face, current_name, name_gender in remaining_trials:
    trialClock.reset()
    stimulus_name_started = False
    stimulus_pic_started = False
    trial_ended = False
    face_onset_time = None
    rt = None
    response_key = None

    # Durations with jitter
    name_duration = random.uniform(0.2, 0.3)
    blank_duration = random.uniform(0.15, 0.25)
    

    # Load face image
    face_full_path = os.path.join(face_path, current_face)
    if os.path.exists(face_full_path):
        Priming_Face.setImage(face_full_path)
    else:
        print("ERROR: Image not found:", face_full_path)

    # Set name
    Priming_Name.setText(current_name)

    # Send combo trigger
    face_gender = "m" if "_m" in current_face else "f"
    combo_code = combo_ids[face_gender + name_gender]
#    outlet.push_sample([combo_code])
    send_trigger(combo_code)
    print("Trial/Combo Trigger sent:", combo_code)

    # Start trial
    continueRoutine = True
    trialClock.reset()

    while continueRoutine:
        t = trialClock.getTime()
        keys = event.getKeys(keyList=['escape'])
        if 'escape' in keys:
            thisExp.saveAsWideText(filename + '.csv')
            thisExp.saveAsPickle(filename)
            thisExp.abort()
            core.quit()

        # NAME DISPLAY
        if t < name_duration:
            Priming_Name.setAutoDraw(True)
            Priming_Face.setAutoDraw(False)
            if not stimulus_name_started:
                NameTrig = name_ids.get(current_name, 0)
#                outlet.push_sample([NameTrig])
                send_trigger(NameTrig)
                print("Name Trigger sent:", NameTrig)
                stimulus_name_started = True

        # BLANK
        elif t < name_duration + blank_duration:
            Priming_Name.setAutoDraw(False)
            Priming_Face.setAutoDraw(False)

        # FACE
        else:
            Priming_Name.setAutoDraw(False)
            Priming_Face.setAutoDraw(True)
            if not stimulus_pic_started:
                face_onset_time = globalClock.getTime()
                ImageTrig = face_ids[current_face]
#                outlet.push_sample([ImageTrig])
                send_trigger(ImageTrig)
                print("Face Trigger sent:", ImageTrig)
                stimulus_pic_started = True

        # Draw response labels
        MAN_3.draw()
        MAN_3_desc.draw()
        WOMAN_3.draw()
        WOMAN_3_desc.draw()
        win.flip()

        # RESPONSE
        keys = event.getKeys(keyList=['j', 'k'], timeStamped=globalClock) # In training we used waitKeys
        if keys and not trial_ended:
            response_key, key_time = keys[0]
            rt = key_time - face_onset_time if face_onset_time else None
            if keys[0][0] == 'j':
                send_trigger(j_key_id)   # arbitrary trigger code for J key
                print(f"Key pressed: {j_key_id}")
            elif keys[0][0] == 'k':
                send_trigger(k_key_id)
                print(f"Key pressed: {k_key_id}")
#            outlet.push_sample([end_trial_code])
            send_trigger(end_trial_code)
#            print(f"Response: {response_key}, RT: {rt:.3f}s")
            trial_ended = True
            continueRoutine = False

        # TIMEOUT (3 s after face onset)
        if (
            face_onset_time  # only count timeout after face has appeared
            and globalClock.getTime() - face_onset_time > 3.0
            and not keys
            and not trial_ended
        ):
            insert_index = random.randint(0, len(remaining_trials))
            remaining_trials.insert(insert_index, (current_face, current_name, name_gender))
#            outlet.push_sample([end_trial_code])
            send_trigger(timeout_code)
            print("End-of-Trial Trigger sent (timeout):", end_trial_code)
            print(f"Timeout! Requeued trial: {current_name}, {current_face}")
            trial_ended = True
            continueRoutine = False
    inter_trial_duration = random.uniform(0.3, 0.5)
    core.wait(inter_trial_duration)

    # === LOG TRIAL DATA ===
    thisExp.addData('Name', current_name)
    thisExp.addData('Face', current_face)
    thisExp.addData('NameGender', name_gender)
    thisExp.addData('ResponseKey', response_key if response_key else 'None')
    thisExp.addData('RT', rt if rt else 'None')
    thisExp.addData('Timeout', 'Yes' if rt is None else 'No')
    thisExp.nextEntry()

# ============ SAVE & QUIT ============
if port is not None:
    port.close()
    print("Serial port closed")
thisExp.saveAsWideText(filename + '.csv')
thisExp.saveAsPickle(filename)
thisExp.abort()
win.close()
core.quit()
