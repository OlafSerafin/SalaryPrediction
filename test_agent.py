import asyncio
from agents import setup_system, call_agent

async def main():
    runner, user_id, session_id = await setup_system()
    print("System is ready.")

    new_user_prompt = """
    Google Company is Hiring: Senior Data Scientist in New York, NY.
    Requirements: Python, AWS, and strong Excel skills.
    """

    response = await call_agent(new_user_prompt, runner, user_id, session_id)
    print("\nAgent answer:\n")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())