# Step 1: Use a Python base image
FROM python:3.11-slim

# Step 2: Set working directory inside the container
WORKDIR /app

# Step 3: Copy requirements.txt to container
COPY requirements.txt .

# Step 4: Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your code
COPY . .

# Step 6: Expose Streamlit port
EXPOSE 8501

# Step 7: Command to run your app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
