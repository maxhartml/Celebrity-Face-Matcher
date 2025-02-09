import os
import logging
import logging.config

# Define the logging configuration dictionary
log_config = {
    'version': 1,
    'disable_existing_loggers': False,
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
            'level': 'DEBUG',  # Adjust to INFO or higher for production if needed
            'formatter': 'default',
        },
        # File handler for the image_processing module logs
        'image_processing_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/image_processing.log',
            'maxBytes': 10 * 1024 * 1024,  # 10 MB per file
            'backupCount': 5,
        },
        # File handler for the matching module logs
        'matching_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'default',
            'filename': 'logs/matching.log',
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
    },
    'loggers': {
        # Logger configuration for the image_processing module
        'app.image_processing': {
            'level': 'DEBUG',
            'handlers': ['image_processing_handler', 'console_handler'],
            'propagate': False,
        },
        # Logger configuration for the matching module
        'app.matching': {
            'level': 'DEBUG',
            'handlers': ['matching_handler', 'console_handler'],
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
    },
    # Root logger configuration—modules without an explicit logger will use this.
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
            # The FileHandler will create the file if it does not already exist.

# Apply the logging configuration
logging.config.dictConfig(log_config)

if __name__ == "__main__":
    # Example usage of the central logging configuration.
    logger = logging.getLogger(__name__)
    logger.info("Central logging configuration has been initialized.")