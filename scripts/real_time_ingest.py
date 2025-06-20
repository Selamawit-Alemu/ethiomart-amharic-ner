from telethon import TelegramClient, events
from dotenv import load_dotenv
load_dotenv('.env')
import os
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')
client = TelegramClient('session', api_id, api_hash)

@client.on(events.NewMessage(chats=['@ZemenExpress', '@sinayelj', '@Leyueqa', '@ethio_brand_collection', '@nevacomputer']))
async def handler(event):
    message = event.message
    # Extract info from message
    print(f"New message from {event.chat_id}: {message.text}")
    # Save message to CSV or database here

client.start()
client.run_until_disconnected()

