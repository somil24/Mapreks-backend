from gevent import monkey
monkey.patch_all()

from app import app

http_server = WSGIServer(('', 5001), app)
http_server.serve_forever()