import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class DynamicGroqLLM:
    def __init__(self, system_prompt: str):
        # Groq client initialize karein apni .env key se
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.system_prompt = system_prompt
        
        # Conversation history maintain karne ke liye list
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]
        
    def generate_response(self, user_text: str) -> str:
        """
        User ka message Groq ko bhejta hai aur context yaad rakhta hai.
        """
        # User ki baat history mein daalein
        self.history.append({"role": "user", "content": user_text})
        
        try:
            # Groq ka sabse fast model use kar rahe hain
            chat_completion = self.client.chat.completions.create(
                messages=self.history,
                model="llama3-8b-8192", # Llama3 bohot fast hai Groq par
                temperature=0.7,
                max_tokens=150 # Chotay answers ke liye taa ke latency kam ho
            )
            
            ai_reply = chat_completion.choices[0].message.content
            
            # AI ka jawab bhi history mein save karein
            self.history.append({"role": "assistant", "content": ai_reply})
            
            return ai_reply
            
        except Exception as e:
            print(f"Groq API Error: {e}")
            return "Maaf karna, mujhe theek se samajh nahi aya. Kya aap repeat karenge?"