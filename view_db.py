import sqlite3
def view_results():
    conn = sqlite3.connect('students.db')
    c = conn.cursor()
    c.execute("SELECT * FROM results")
    rows = c.fetchall()
    conn.close()

    for row in rows:
        print(row)


if __name__ == "__main__":
    view_results()