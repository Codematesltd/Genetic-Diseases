
from supabase import create_client
from os import getenv
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = getenv('SUPABASE_URL')
supabase_key = getenv('SUPABASE_KEY')

if not supabase_url or not supabase_key:
    raise ValueError("Missing Supabase credentials. Check your .env file.")

supabase = create_client(supabase_url, supabase_key)

class User:
    @staticmethod
    def create_user(email, password, username, first_name, last_name):
        try:
            # Create auth user - Supabase handles password hashing
            auth_response = supabase.auth.sign_up({
                "email": email,
                "password": password
            })
            
            if auth_response.user:
                user_data = {
                    "id": auth_response.user.id,
                    "username": username,
                    "email": email,
                    "first_name": first_name,
                    "last_name": last_name
                }
                
                # Insert into users table
                supabase.table('users').insert(user_data).execute()
                return auth_response.user
                
        except Exception as e:
            raise Exception(f"Error creating user: {str(e)}")
    
    @staticmethod
    def login(email, password):
        try:
            auth_response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if auth_response.user:
                return auth_response
            return None
        except Exception as e:
            raise Exception(f"Error logging in: {str(e)}")
    
    @staticmethod
    def get_user_by_id(user_id):
        try:
            response = supabase.table('users').select("*").eq('id', user_id).execute()
            if response.data:
                return response.data[0]
            return None
        except Exception as e:
            raise Exception(f"Error fetching user: {str(e)}")
    
    @staticmethod
    def logout():
        try:
            supabase.auth.sign_out()
        except Exception as e:
            raise Exception(f"Error logging out: {str(e)}")
            
    @staticmethod
    def get_current_user():
        try:
            user = supabase.auth.get_user()
            if user:
                return User.get_user_by_id(user.id)
            return None
        except Exception:
            return None
