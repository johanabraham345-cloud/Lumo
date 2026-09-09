# LUMO

## Why I built this

A lot of kids with learning disabilities don't do well with how school normally teaches things — sit still, read the text, listen to the teacher. If a concept doesn't land through those channels there's often no backup plan, especially in a classroom where one teacher can't give everyone individual attention. Abstract ideas stay abstract, and kids who need to see and touch something to understand it end up disengaging.

LUMO is my attempt at giving these kids another way in. It's a small robot that teaches through voice, sight, and physical movement instead of just talking at someone or handing them a worksheet — so a lesson becomes something you do with your hands, not just something you're told.

![IMG_5251 (1)](https://github.com/user-attachments/assets/12c0066d-d809-4f16-9a7a-2e29dafa2b82)
![IMG_5334](https://github.com/user-attachments/assets/f8958f99-30ca-4c31-a9c2-0ba749172b07)
![IMG_5252 (1)](https://github.com/user-attachments/assets/23067795-520b-4d24-81a9-081e86b45c33)

**How to use**

Power on the system and launch the main program on the Raspberry Pi. Once it's running, LUMO starts listening through the mic — just talk to it like you would a person, answer its questions or give it a command. It converts what you said, works out a response, and speaks back through the speaker.

Put an object where the camera can see it and LUMO will recognize it and react. Depending on the activity, the arm picks it up and sorts it into one of the two containers. The display keeps the session going with questions, feedback, and prompts so it's clear what to do next.

⚙️ **How it works**

**1. Voice interaction (NLP)**
The mic picks up what's said, a local model turns it into text, that gets processed to figure out a response, and the reply comes back out through the speaker.

![IMG_5237](https://github.com/user-attachments/assets/29a48fe0-d44a-42c9-81a6-e308d8df2304)

**2. Vision system**
A camera handles the seeing side of things — colours (red, blue, green, yellow), objects, and faces, with face detection used to keep LUMO tracking and engaged with whoever it's working with.

![IMG_5385](https://github.com/user-attachments/assets/ab7e8077-e3f6-42be-91b9-862715312239)
![Screenshot 2026-01-15 161725](https://github.com/user-attachments/assets/d55a9d50-b2a3-43c1-a5d7-45389a3a7466)
![Screenshot 2026-01-15 162325](https://github.com/user-attachments/assets/9840a6d9-096d-4a09-a8e9-9e2b9d8ef768)

**3. Robotic arm**
Takes the commands and does the physical part — picking things up and sorting them. It uses two containers so the sorting is something the kid can actually watch happen, not just something LUMO tells them about.

![IMG_5252 (1)](https://github.com/user-attachments/assets/8e3feaf6-a885-48f1-8290-323d15b6e8a1)
![IMG_5267](https://github.com/user-attachments/assets/8bbc13d5-8128-4e81-9476-659acb6a799d)

**4. Interactive feedback**
The screen carries the rest of the interaction — questions, feedback, and prompts that keep the activity moving and give the kid something to focus on while they think.

**Circuit Design**

![Lumo Circuit Schematic (1)](https://github.com/user-attachments/assets/8243c9d4-496f-42c4-a157-7e07ae96099a)

**Credits**

3D modelling: done in Fusion 360 — the arm structure and the custom parts.
Software: Python, using libraries for AI, computer vision, and hardware control.
