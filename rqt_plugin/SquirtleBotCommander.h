#ifndef SquirtleBotCommander_h
#define SquirtleBotCommander_h_h

#include <ros/ros.h>
#include <rqt_gui_cpp/plugin.h>

#include <std_msgs/String.h>

#include <QList>
#include <QTime>
#include <QString>
#include <QWidget>

#include "ui_SquirtleBotCommander.h"

  
class SquirtleBotCommander : public rqt_gui_cpp::Plugin
{
    Q_OBJECT
    public:
        SquirtleBotCommander();
        ~SquirtleBotCommander();
        
        virtual void initPlugin(qt_gui_cpp::PluginContext& context);

        virtual void shutdownPlugin();

        virtual void saveSettings(qt_gui_cpp::Settings& plugin_settings, 
                              qt_gui_cpp::Settings& instance_settings) const;
                              
        virtual void restoreSettings(const qt_gui_cpp::Settings& plugin_settings, 
                                 const qt_gui_cpp::Settings& instance_settings);

    protected slots:
        void on_exploreBut_clicked();
        void on_orderBut_clicked();
    
    protected:
        ros::Publisher    m_publisher_exp;
        ros::Publisher    m_publisher_ord;
        
    protected:
        Ui::SquirtleBotCommanderPanel m_ui;
        QWidget*          m_widget;
  };


#endif // SquirtleBotCommander_h
