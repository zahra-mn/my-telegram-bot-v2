import os
import logging
import telebot
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
HF_TOKEN = os.environ.get('HUGGINGFACE_API_KEY')

if not BOT_TOKEN or not HF_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN and HUGGINGFACE_API_KEY must be set")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

bot = telebot.TeleBot(BOT_TOKEN )

def get_hf_response(user_message):
    payload = {
        "inputs": f"<s>[INST] {user_message} [/INST]",
        "parameters": {"max_new_tokens": 500, "temperature": 0.7, "return_full_text": False}
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text'].strip()
        logger.error(f"Unexpected API response format: {result}")
        return "پاسخ غیرمنتظره‌ای از مدل دریافت شد."
    except requests.exceptions.RequestException as e:
        error_text = f"Error: {e.response.text if e.response else e}"
        if "is currently loading" in error_text:
            return "مدل در حال بارگذاری است، لطفاً چند دقیقه دیگر دوباره تلاش کنید."
        logger.error(f"Error calling Hugging Face API: {error_text}")
        return "متاسفانه مشکلی در ارتباط با Hugging Face پیش آمد."

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "سلام! من با مدل Mistral از Hugging Face کار می‌کنم.")

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    bot.send_chat_action(message.chat.id, 'typing')
    response = get_hf_response(message.text)
    bot.reply_to(message, response)

logger.info("Bot is running...")
bot.infinity_polling()
