import sqlite3
import bcrypt

DB_NAME = 'users.db'

def get_connection():
    return sqlite3.connect(DB_NAME, timeout=10)

def create_tables():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_verified INTEGER DEFAULT 0,
                otp TEXT,
                role TEXT DEFAULT 'user'
            )
        ''')
        conn.commit()

def fix_passwords():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT id, password FROM users")
    rows = c.fetchall()
    for user_id, pwd in rows:
        pwd_str = pwd.decode('utf-8') if isinstance(pwd, bytes) else pwd

        # Only rehash if NOT a valid bcrypt hash (starts with $2a$, $2b$, or $2y$ and length ~60)
        if not (pwd_str.startswith('$2a$') or pwd_str.startswith('$2b$') or pwd_str.startswith('$2y$')) or len(pwd_str) < 50:
            print(f"Rehashing password for user id {user_id}")
            hashed = bcrypt.hashpw(pwd_str.encode('utf-8'), bcrypt.gensalt())
            hashed_str = hashed.decode('utf-8')
            c.execute("UPDATE users SET password=? WHERE id=?", (hashed_str, user_id))
    conn.commit()
    conn.close()

def add_user(email, password, otp, role='user'):
    try:
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        hashed_str = hashed.decode('utf-8')
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (email, password, is_verified, otp, role) VALUES (?, ?, 0, ?, ?)",
                      (email, hashed_str, otp, role))
            conn.commit()
        return True
    except Exception as e:
        print("DB Error in add_user():", e)
        return False

def get_user_by_email(email):
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email=?", (email,))
            return c.fetchone()
    except Exception as e:
        print("DB Error (get_user_by_email):", e)
        return None

def authenticate_user(email, password):
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email=? AND is_verified=1", (email,))
            user = c.fetchone()
            if user:
                stored_hashed_password = user[2]
                if isinstance(stored_hashed_password, bytes):
                    stored_hashed_password = stored_hashed_password.decode('utf-8')

                print("Stored hash:", repr(stored_hashed_password))

                if bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
                    return {
                        "id": user[0],
                        "email": user[1],
                        "role": user[5]
                    }
    except Exception as e:
        print("DB Error (authenticate_user):", e)
    return None

def verify_user(email, code):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT * FROM users WHERE email = ? AND otp = ?", (email, code))
        user = c.fetchone()
        if user:
            c.execute("UPDATE users SET is_verified = 1, otp = NULL WHERE email = ?", (email,))
            conn.commit()
            return True
        return False
    finally:
        conn.close()

def get_all_users():
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT email, is_verified, role FROM users")
            return c.fetchall()
    except Exception as e:
        print("DB Error (get_all_users):", e)
        return []

def delete_user_by_email(email):
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM users WHERE email=?", (email,))
            conn.commit()
    except Exception as e:
        print("DB Error (delete_user_by_email):", e)



def set_reset_code(email, code):
    """Store the reset code (OTP) in the user's record."""
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("UPDATE users SET otp=? WHERE email=?", (code, email))
            conn.commit()
        return True
    except Exception as e:
        print("DB Error (set_reset_code):", e)
        return False

def verify_reset_code(email, code):
    """Check if the provided reset code matches the stored one."""
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT otp FROM users WHERE email=?", (email,))
            row = c.fetchone()
            if row and row[0] == code:
                return True
    except Exception as e:
        print("DB Error (verify_reset_code):", e)
    return False

def update_user_password(email, new_password):
    """Update user password with bcrypt hashing, clear the OTP."""
    try:
        hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        hashed_str = hashed.decode('utf-8')
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("UPDATE users SET password=?, otp=NULL WHERE email=?", (hashed_str, email))
            conn.commit()
        return True
    except Exception as e:
        print("DB Error (update_user_password):", e)
        return False


# --- Run once to create table and fix passwords ---
if __name__ == "__main__":
    create_tables()
    fix_passwords()
    print("Database ready and passwords fixed if needed.")
