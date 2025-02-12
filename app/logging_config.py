import os
import logging
import logging.config

# Define the logging configuration dictionary
log_config = {
    'version': 1,
    'disable_existing_loggers': False,  # Do not disable existing loggers
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        # Console handler for real-time logging output
        'console_handler': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',  # Change to INFO or higher in production if needed
            'formatter': 'default',
        },
        # File handler for the image_processing module logs
        'image_processing_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/image_processing.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
        },
        # File handler for the orchestration module logs
        'orchestration_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/orchestration.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
        },
        # File handler for the explanation module logs
        'explanation_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/explanation.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
        },
        # File handler for the API module logs
        'api_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/api.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
        },
        # File handler for the vector_store module logs
        'vector_store_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/vector_store.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
        },
        # File handler for the data module logs
        'data_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/data.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
        },
        # File handler for the query module logs
        'query_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/query.log',
            'maxBytes': 10 * 1024 * 1024,
            'backupCount': 5,
        },
    },
    'loggers': {
        # Logger configuration for the image_processing module
        'app.image_processing': {
            'level': 'DEBUG',
            'handlers': ['image_processing_handler', 'console_handler'],
            'propagate': False,
        },
        # Logger configuration for the orchestration module
        'app.orchestration': {
            'level': 'DEBUG',
            'handlers': ['orchestration_handler', 'console_handler'],
            'propagate': False,
        },
        # Logger configuration for the explanation module
        'app.explanation': {
            'level': 'DEBUG',
            'handlers': ['explanation_handler', 'console_handler'],
            'propagate': False,
        },
        # Logger configuration for the API module
        'app.api': {
            'level': 'DEBUG',
            'handlers': ['api_handler', 'console_handler'],
            'propagate': False,
        },
        # Logger configuration for the vector_store module
        'app.vector_store': {
            'level': 'DEBUG',
            'handlers': ['vector_store_handler', 'console_handler'],
            'propagate': False,
        },
        # Logger configuration for the data module
        'app.data': {
            'level': 'DEBUG',
            'handlers': ['data_handler', 'console_handler'],
            'propagate': False,
        },
        # Logger configuration for the query module
        'app.query': {
            'level': 'DEBUG',
            'handlers': ['query_handler', 'console_handler'],
            'propagate': False,
        },
        # Configure pymongo to only log warnings or errors.
        'pymongo': {
            'level': 'WARNING',
            'handlers': ['console_handler'],
            'propagate': False,
        },
    },
    # Root logger configuration for modules without an explicit logger.
    'root': {
        'level': 'DEBUG',
        'handlers': ['console_handler']
    }
}

# Ensure the log directories exist before configuring the logging system.
for handler in log_config['handlers'].values():
    filename = handler.get('filename')
    if filename:
        log_dir = os.path.dirname(filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

# Apply the logging configuration
logging.config.dictConfig(log_config)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Central logging configuration has been initialized.")