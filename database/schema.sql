-- ExoAI Hunter Database Schema
-- PostgreSQL database for storing exoplanet data, predictions, and analytics

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Create database user and permissions
-- CREATE USER exoai_user WITH PASSWORD 'secure_password_here';
-- GRANT ALL PRIVILEGES ON DATABASE exoai_hunter TO exoai_user;

-- ==============================================
-- CORE TABLES
-- ==============================================

-- Missions table
CREATE TABLE missions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    full_name VARCHAR(200) NOT NULL,
    launch_date DATE,
    end_date DATE,
    description TEXT,
    data_cadence INTERVAL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert mission data
INSERT INTO missions (name, full_name, launch_date, end_date, description, data_cadence) VALUES
('kepler', 'Kepler Space Telescope', '2009-03-07', '2017-10-30', 'NASA space telescope designed to discover Earth-size planets', '29.4 minutes'),
('k2', 'K2 Extended Mission', '2014-05-30', '2018-10-30', 'Extended mission of Kepler telescope', '29.4 minutes'),
('tess', 'Transiting Exoplanet Survey Satellite', '2018-04-18', NULL, 'NASA space telescope searching for exoplanets', '2 minutes');

-- Host stars table
CREATE TABLE host_stars (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    ra DOUBLE PRECISION, -- Right ascension in degrees
    dec DOUBLE PRECISION, -- Declination in degrees
    magnitude REAL, -- Apparent magnitude
    temperature REAL, -- Effective temperature in Kelvin
    radius REAL, -- Stellar radius in solar radii
    mass REAL, -- Stellar mass in solar masses
    distance REAL, -- Distance in parsecs
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for coordinate searches
CREATE INDEX idx_host_stars_coordinates ON host_stars USING GIST (ll_to_earth(dec, ra));
CREATE INDEX idx_host_stars_name ON host_stars USING GIN (name gin_trgm_ops);

-- Exoplanet objects table
CREATE TABLE exoplanet_objects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    object_id VARCHAR(50) NOT NULL UNIQUE, -- KOI-326.01, TOI-715.01, etc.
    mission_id INTEGER REFERENCES missions(id),
    host_star_id INTEGER REFERENCES host_stars(id),
    
    -- Object properties
    name VARCHAR(100), -- Common name if available
    disposition VARCHAR(20) NOT NULL, -- CONFIRMED, CANDIDATE, FALSE_POSITIVE
    
    -- Orbital parameters
    period REAL, -- Orbital period in days
    period_err1 REAL, -- Upper error on period
    period_err2 REAL, -- Lower error on period
    
    -- Transit parameters
    depth REAL, -- Transit depth in ppm
    depth_err1 REAL,
    depth_err2 REAL,
    duration REAL, -- Transit duration in hours
    duration_err1 REAL,
    duration_err2 REAL,
    
    -- Physical parameters
    radius REAL, -- Planet radius in Earth radii
    radius_err1 REAL,
    radius_err2 REAL,
    mass REAL, -- Planet mass in Earth masses
    mass_err1 REAL,
    mass_err2 REAL,
    equilibrium_temp REAL, -- Equilibrium temperature in Kelvin
    
    -- Discovery information
    discovery_year INTEGER,
    discovery_method VARCHAR(50) DEFAULT 'Transit',
    
    -- Metadata
    data_quality_flags TEXT[],
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX idx_exoplanet_objects_mission ON exoplanet_objects(mission_id);
CREATE INDEX idx_exoplanet_objects_disposition ON exoplanet_objects(disposition);
CREATE INDEX idx_exoplanet_objects_discovery_year ON exoplanet_objects(discovery_year);
CREATE INDEX idx_exoplanet_objects_name ON exoplanet_objects USING GIN (name gin_trgm_ops);

-- Light curve data table (for storing processed light curves)
CREATE TABLE light_curves (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    object_id UUID REFERENCES exoplanet_objects(id) ON DELETE CASCADE,
    mission_id INTEGER REFERENCES missions(id),
    
    -- Light curve metadata
    data_type VARCHAR(20) NOT NULL, -- RAW, PROCESSED, DETRENDED
    cadence INTERVAL,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    data_points INTEGER NOT NULL,
    
    -- Data quality metrics
    rms_noise REAL,
    snr REAL, -- Signal-to-noise ratio
    completeness REAL, -- Data completeness percentage
    
    -- File storage information
    file_path VARCHAR(500), -- Path to stored light curve file
    file_format VARCHAR(10) DEFAULT 'FITS', -- FITS, CSV, JSON
    file_size BIGINT, -- File size in bytes
    
    -- Processing information
    processing_pipeline VARCHAR(100),
    processing_version VARCHAR(20),
    processed_at TIMESTAMP WITH TIME ZONE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_light_curves_object ON light_curves(object_id);
CREATE INDEX idx_light_curves_mission ON light_curves(mission_id);
CREATE INDEX idx_light_curves_data_type ON light_curves(data_type);

-- ==============================================
-- AI/ML PREDICTION TABLES
-- ==============================================

-- AI model versions table
CREATE TABLE ai_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- CNN, TRANSFORMER, ENSEMBLE
    architecture_details JSONB,
    
    -- Training information
    training_dataset VARCHAR(100),
    training_start_date TIMESTAMP WITH TIME ZONE,
    training_end_date TIMESTAMP WITH TIME ZONE,
    epochs INTEGER,
    
    -- Performance metrics
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    
    -- Model file information
    model_file_path VARCHAR(500),
    model_file_size BIGINT,
    
    -- Status and metadata
    status VARCHAR(20) DEFAULT 'TRAINING', -- TRAINING, ACTIVE, DEPRECATED
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(name, version)
);

-- Predictions table
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    object_id UUID REFERENCES exoplanet_objects(id),
    model_id INTEGER REFERENCES ai_models(id),
    light_curve_id UUID REFERENCES light_curves(id),
    
    -- Prediction results
    predicted_class VARCHAR(20) NOT NULL, -- CONFIRMED, CANDIDATE, FALSE_POSITIVE
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Probability scores for each class
    prob_confirmed REAL NOT NULL CHECK (prob_confirmed >= 0 AND prob_confirmed <= 1),
    prob_candidate REAL NOT NULL CHECK (prob_candidate >= 0 AND prob_candidate <= 1),
    prob_false_positive REAL NOT NULL CHECK (prob_false_positive >= 0 AND prob_false_positive <= 1),
    
    -- Uncertainty and quality metrics
    uncertainty REAL, -- Model uncertainty estimate
    entropy REAL, -- Prediction entropy
    
    -- Processing metadata
    processing_time_ms INTEGER, -- Processing time in milliseconds
    batch_id UUID, -- For batch processing tracking
    
    -- Validation (if available)
    true_class VARCHAR(20), -- Actual disposition if known
    is_correct BOOLEAN, -- Whether prediction matches true class
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for prediction queries
CREATE INDEX idx_predictions_object ON predictions(object_id);
CREATE INDEX idx_predictions_model ON predictions(model_id);
CREATE INDEX idx_predictions_class ON predictions(predicted_class);
CREATE INDEX idx_predictions_confidence ON predictions(confidence DESC);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);

-- ==============================================
-- ANALYTICS AND REPORTING TABLES
-- ==============================================

-- Model performance tracking
CREATE TABLE model_performance_logs (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES ai_models(id),
    
    -- Performance metrics
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    roc_auc REAL,
    
    -- Dataset information
    test_dataset_size INTEGER,
    validation_method VARCHAR(50), -- K_FOLD, HOLDOUT, TEMPORAL
    
    -- Mission-specific performance
    kepler_accuracy REAL,
    k2_accuracy REAL,
    tess_accuracy REAL,
    
    -- Timestamp
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- User sessions and activity tracking
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(100) NOT NULL,
    user_agent TEXT,
    ip_address INET,
    
    -- Activity metrics
    predictions_made INTEGER DEFAULT 0,
    files_uploaded INTEGER DEFAULT 0,
    objects_viewed INTEGER DEFAULT 0,
    
    -- Session timing
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE
);

-- System metrics and monitoring
CREATE TABLE system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metric_unit VARCHAR(20),
    tags JSONB, -- Additional metadata as JSON
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_system_metrics_name_time ON system_metrics(metric_name, timestamp DESC);

-- ==============================================
-- DATA VALIDATION AND CONSTRAINTS
-- ==============================================

-- Add constraint to ensure probability scores sum to ~1
ALTER TABLE predictions ADD CONSTRAINT check_probability_sum 
    CHECK (ABS((prob_confirmed + prob_candidate + prob_false_positive) - 1.0) < 0.001);

-- Add constraint for valid dispositions
ALTER TABLE exoplanet_objects ADD CONSTRAINT check_disposition 
    CHECK (disposition IN ('CONFIRMED', 'CANDIDATE', 'FALSE_POSITIVE', 'NOT_DISPOSITIONED'));

-- Add constraint for valid model status
ALTER TABLE ai_models ADD CONSTRAINT check_model_status 
    CHECK (status IN ('TRAINING', 'ACTIVE', 'DEPRECATED', 'FAILED'));

-- ==============================================
-- VIEWS FOR COMMON QUERIES
-- ==============================================

-- View for latest predictions with object details
CREATE VIEW latest_predictions AS
SELECT 
    p.id,
    p.object_id,
    eo.object_id as object_identifier,
    eo.name,
    eo.disposition as true_disposition,
    p.predicted_class,
    p.confidence,
    p.prob_confirmed,
    p.prob_candidate,
    p.prob_false_positive,
    p.uncertainty,
    p.processing_time_ms,
    m.name as mission_name,
    am.name as model_name,
    am.version as model_version,
    p.created_at
FROM predictions p
JOIN exoplanet_objects eo ON p.object_id = eo.id
JOIN missions m ON eo.mission_id = m.id
JOIN ai_models am ON p.model_id = am.id
WHERE p.created_at = (
    SELECT MAX(created_at) 
    FROM predictions p2 
    WHERE p2.object_id = p.object_id
);

-- View for model performance summary
CREATE VIEW model_performance_summary AS
SELECT 
    am.id,
    am.name,
    am.version,
    am.model_type,
    am.accuracy,
    am.precision,
    am.recall,
    am.f1_score,
    COUNT(p.id) as total_predictions,
    COUNT(CASE WHEN p.is_correct = true THEN 1 END) as correct_predictions,
    AVG(p.confidence) as avg_confidence,
    AVG(p.processing_time_ms) as avg_processing_time,
    am.created_at,
    am.status
FROM ai_models am
LEFT JOIN predictions p ON am.id = p.model_id
GROUP BY am.id, am.name, am.version, am.model_type, am.accuracy, 
         am.precision, am.recall, am.f1_score, am.created_at, am.status;

-- ==============================================
-- FUNCTIONS AND TRIGGERS
-- ==============================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_missions_updated_at BEFORE UPDATE ON missions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_host_stars_updated_at BEFORE UPDATE ON host_stars
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_exoplanet_objects_updated_at BEFORE UPDATE ON exoplanet_objects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_light_curves_updated_at BEFORE UPDATE ON light_curves
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ai_models_updated_at BEFORE UPDATE ON ai_models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate model accuracy
CREATE OR REPLACE FUNCTION calculate_model_accuracy(model_id_param INTEGER)
RETURNS REAL AS $$
DECLARE
    accuracy_result REAL;
BEGIN
    SELECT 
        CASE 
            WHEN COUNT(*) = 0 THEN NULL
            ELSE COUNT(CASE WHEN is_correct = true THEN 1 END)::REAL / COUNT(*)::REAL
        END
    INTO accuracy_result
    FROM predictions 
    WHERE model_id = model_id_param AND true_class IS NOT NULL;
    
    RETURN accuracy_result;
END;
$$ LANGUAGE plpgsql;

-- ==============================================
-- SAMPLE DATA INSERTION
-- ==============================================

-- Insert sample host stars
INSERT INTO host_stars (name, ra, dec, magnitude, temperature, radius, mass, distance) VALUES
('Kepler-22', 290.857, 47.883, 11.664, 5518, 0.979, 0.970, 190),
('TOI-715', 97.631, -31.293, 12.43, 3450, 0.374, 0.374, 42),
('K2-18', 172.560, 7.588, 13.457, 3457, 0.411, 0.359, 38);

-- Insert sample AI models
INSERT INTO ai_models (name, version, model_type, accuracy, precision, recall, f1_score, status, description) VALUES
('ExoAI-CNN', '1.0', 'CNN', 0.962, 0.945, 0.951, 0.948, 'ACTIVE', 'Convolutional Neural Network with attention mechanisms'),
('ExoAI-Transformer', '1.0', 'TRANSFORMER', 0.958, 0.941, 0.949, 0.945, 'ACTIVE', 'Transformer-based sequence model'),
('ExoAI-Ensemble', '1.0', 'ENSEMBLE', 0.973, 0.965, 0.958, 0.961, 'ACTIVE', 'Ensemble of CNN and Transformer models');

-- Grant permissions
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO exoai_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO exoai_user;

-- Create indexes for full-text search
CREATE INDEX idx_exoplanet_objects_fulltext ON exoplanet_objects 
    USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(object_id, '')));

-- Final comment
COMMENT ON DATABASE exoai_hunter IS 'ExoAI Hunter - AI-Powered Exoplanet Detection Platform Database';
COMMENT ON TABLE exoplanet_objects IS 'Core table storing exoplanet objects from NASA missions';
COMMENT ON TABLE predictions IS 'AI model predictions for exoplanet classification';
COMMENT ON TABLE ai_models IS 'Machine learning model metadata and performance metrics';
