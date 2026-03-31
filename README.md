💡 Why LUMO was built

**Traditional learning methods often don’t work well for children who need more visual, hands-on, and interactive experiences. Many students struggle with:**

Lack of individual attention
Difficulty understanding abstract concepts
Low engagement in passive learning

**LUMO was created to make learning:**

More interactive (through physical actions)
More accessible (simple voice-based interaction)
More engaging (visual + audio + movement together)

**This project was created to support children with learning disabilities who often struggle with traditional teaching methods. LUMO aims to make learning more engaging and easier to understand by combining visual, audio, and physical interaction, especially in situations where individual attention is limited.**
![IMG_5251 (1)](https://github.com/user-attachments/assets/12c0066d-d809-4f16-9a7a-2e29dafa2b82)
![IMG_5334](https://github.com/user-attachments/assets/f8958f99-30ca-4c31-a9c2-0ba749172b07)
![IMG_5252 (1)](https://github.com/user-attachments/assets/23067795-520b-4d24-81a9-081e86b45c33)


▶️ **How to use**

Power on the system and launch the main program on the Raspberry Pi. Once running, LUMO enters listening mode through the microphone. Speak clearly to give commands or respond to its questions. The system processes your voice input, generates a response, and replies through the speaker.

Place objects within the camera’s view so LUMO can detect and recognise them. Based on the interaction, the robotic arm will move to pick, place, or sort objects accordingly. Follow the prompts shown on the display, which guide the activity with questions, feedback, and instructions to keep the interaction smooth and engaging.

⚙️ **How it works**

1. **Voice Interaction (NLP)**
The system listens using a microphone
Converts speech to text using local models
Processes input using AI
Responds with voice output
![IMG_5237](https://github.com/user-attachments/assets/29a48fe0-d44a-42c9-81a6-e308d8df2304)

2. **Vision System**
Uses a camera to detect:
Colours (red, blue, green, yellow)
Objects
Faces (for tracking and engagement)
![IMG_5385](https://github.com/user-attachments/assets/ab7e8077-e3f6-42be-91b9-862715312239)
![Screenshot 2026-01-15 161725](https://github.com/user-attachments/assets/d55a9d50-b2a3-43c1-a5d7-45389a3a7466)
![Screenshot 2026-01-15 162325](https://github.com/user-attachments/assets/9840a6d9-096d-4a09-a8e9-9e2b9d8ef768)

3. **Robotic Arm**
Responds to commands
Picks and sorts objects
Uses two containers to visually represent sorting
![IMG_5252 (1)](https://github.com/user-attachments/assets/8e3feaf6-a885-48f1-8290-323d15b6e8a1)
![IMG_5267](https://github.com/user-attachments/assets/8bbc13d5-8128-4e81-9476-659acb6a799d)

4. **Interactive Feedback**
Screen shows:
Questions
Feedback
Prompts and learning cues

**Circuit Design**
![Lumo Circuit Schematic (1)](https://github.com/user-attachments/assets/8243c9d4-496f-42c4-a157-7e07ae96099a)

📌 Credits
3D Modelling: Designed using Fusion 360 for creating the robotic arm structure and custom components.
Software Development: Implemented using Python with libraries for AI, computer vision, and hardware control.

