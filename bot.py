import asyncio
import logging
import httpx
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from config.constants import settings


logging.basicConfig(level=logging.DEBUG)


bot = Bot(token=settings.bot.TOKEN)
dp = Dispatcher()

# Авторизация бота в FastAPI сервисе
ACCESS_TOKEN = None
async def get_jwt_token():
    global ACCESS_TOKEN
    
    login_url = f"{settings.bot.API_URL}{settings.app.PUBLIC_URLS['login']}"

    async with httpx.AsyncClient() as client:
        try:
            logging.info(f"Бот пытается войти в API: {login_url}")
            response = await client.post(
                login_url,
                json={                                 
                    "email": settings.bot.API_EMAIL,  
                    "password": settings.bot.API_PASSWORD
                }
            )
            response.raise_for_status()
            tokens = response.json()
            ACCESS_TOKEN = tokens["access_token"]
            logging.info("✅ Бот успешно авторизовался в API!")
        except Exception as e:
            logging.error(f"❌ Ошибка авторизации бота: {e}")
            logging.error("Проверьте, запущен ли сервер и создан ли пользователь бота.")
            ACCESS_TOKEN = None

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "Hello! Customer support here.\n"
        "How can I help you?"
    )

@dp.message(F.text)
async def handle_text(message: types.Message):
    if not ACCESS_TOKEN:
        await message.answer("⚠️ Бот не подключен к API. Пробую переподключиться...")
        await get_jwt_token()
        if not ACCESS_TOKEN:
            await message.answer("Не удалось подключиться к серверу.")
            return

    predict_url = f"{settings.bot.API_URL}{settings.app.URL_PREFIX}/dl/intent/forward"
    
    payload = {
        "message": message.text, 
    }

    # 2 попытки: первая + повторная после обновления токена
    max_attempts = 2
    for attempt in range(max_attempts):
        if not ACCESS_TOKEN:
            await get_jwt_token()
            if not ACCESS_TOKEN:
                await message.answer("Не удалось подключиться к серверу.")
                return

        headers = {
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(predict_url, json=payload, headers=headers)
                
                if response.status_code == 200 or response.status_code == 201:
                    result = response.json()
                    
                    intent = result.get("intents", "Не определено")
                    confidence = result.get("time_taken", 0) 

                    text_response = (
                        f"📩 Request: <i>{payload['message']}</i>\n"
                        f"🏷 <b>Intent</b>: <code>{intent}</code>"
                    )
                    await message.answer(text_response, parse_mode="HTML")
                    return  
                
                elif response.status_code == 401:
                    logging.info("🔄 Токен устарел. Обновляю...")
                    await get_jwt_token()
                    if ACCESS_TOKEN and attempt < max_attempts - 1:
                        continue
                    else:
                        await message.answer("Не удалось обновить токен доступа.")
                        return
                
                else:
                    # Ошибка (404, 500 и т.д.)
                    await message.answer(f"Ошибка API: {response.status_code}\n{response.text}")
                    return

            except httpx.RequestError as e:
                await message.answer(f"Ошибка соединения с сервером: {e}")
                return

async def main():
    await get_jwt_token()
    
    print("Бот запущен...")
    await dp.start_polling(bot)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('Бот отключён')