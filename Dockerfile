FROM python:3.12-slim
WORKDIR /app
COPY python/ /app/python/
COPY fivebit/ /app/fivebit/
COPY c/ /app/c/
COPY examples/ /app/examples/
RUN pip install --break-system-packages websockets
RUN cd /app/c && gcc -O2 -shared -fPIC -o libfivebit.so fivebit_lib.c
ENV PYTHONPATH=/app:/app/python
EXPOSE 8080 8081
CMD python3 -c "from fivebit.api.server import APIServer; APIServer('/data', {'name':'records','fields':['value']}, port=8080).start(True)" & python3 -c "from fivebit.api.realtime import RealtimeServer; import asyncio; asyncio.run(RealtimeServer('/data', 8081).start_ws())" & wait
