from sqlalchemy import (
    Table,
    Column,
    String,
    Enum,
    DateTime,
    ForeignKey,
    create_engine,
    MetaData,
    UniqueConstraint,
    select,
    delete,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
import bcrypt
import os
from dotenv import load_dotenv
from enum import StrEnum, auto
from datetime import datetime, timezone


class userType(StrEnum):
    admin = auto()
    owner = auto()
    viewer = auto()


# ------------------------------------------------------------
USERNAME = ""
PASSWORD = ""
# e.g. ROLES = {userType.admin.value: "installation1", userType.owner.value: "installation2"}
ROLES = {userType.owner.value: "spruce"}
DELETE_EXISTING_TABLES = False
# ------------------------------------------------------------

load_dotenv()
database_url = os.getenv("GBO_DB_URL_NO_ASYNC")
engine_gbo = create_engine(database_url)

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("username", String, nullable=False),
    Column("hashed_password", String, nullable=False),
    Column("last_login", DateTime, nullable=True),
    UniqueConstraint("username", name="uq_users_username"),
)

user_roles = Table(
    "user_roles",
    metadata,
    Column(
        "username",
        String,
        ForeignKey("users.username", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("role", Enum(userType, name="user_type_enum", native_enum=False), nullable=False),
    Column("installation", String, nullable=True),
    UniqueConstraint("username", "installation", name="uq_user_roles_username_installation"),
)


def get_password_hash(password):
    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    return hashed.decode("utf-8")

Session = sessionmaker(bind=engine_gbo)


def _confirm_replace_existing_user(session):
    """If USERNAME exists, prompt to delete or cancel. Returns False to abort."""
    exists = session.execute(
        select(users.c.username).where(users.c.username == USERNAME)
    ).first()
    if exists is None:
        return True

    prompt = (
        f'User "{USERNAME}" already exists. Delete this user from users and user_roles '
        "and continue with the new details? [y/N]: "
    )
    answer = input(prompt).strip().lower()
    if answer not in ("y", "yes"):
        print("Cancelled.")
        return False

    session.execute(delete(users).where(users.c.username == USERNAME))
    return True


def create_user():
    if DELETE_EXISTING_TABLES:
        metadata.drop_all(engine_gbo, tables=[user_roles, users], checkfirst=True)
        metadata.create_all(engine_gbo, tables=[users, user_roles], checkfirst=True)

    hashed_password = get_password_hash(PASSWORD)
    now = datetime.now(timezone.utc)

    session = Session()

    if not _confirm_replace_existing_user(session):
        session.close()
        return

    user_row = {
        "username": USERNAME,
        "hashed_password": hashed_password,
        "last_login": now,
    }
    stmt_users = insert(users).values(**user_row)
    upsert_users = stmt_users.on_conflict_do_update(
        index_elements=["username"],
        set_={key: getattr(stmt_users.excluded, key) for key in user_row.keys()},
    )
    session.execute(upsert_users)

    for role, installation in ROLES.items():
        role_row = {
            "username": USERNAME,
            "role": role,
            "installation": installation,
        }
        stmt_roles = insert(user_roles).values(**role_row)
        upsert_roles = stmt_roles.on_conflict_do_update(
            index_elements=["username", "installation"],
            set_={key: getattr(stmt_roles.excluded, key) for key in role_row.keys()},
        )
        session.execute(upsert_roles)

    session.commit()
    print("User upserted successfully")
    session.close()


if __name__ == "__main__":
    create_user()
