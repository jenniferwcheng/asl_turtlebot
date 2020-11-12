#include "SquirtleBotCommander.h"

#include <pluginlib/class_list_macros.h>
#include <ros/master.h>

#include <std_msgs/String.h>
#include <QComboBox>
#include <QPushButton>
#include <QTextEdit>
#include <QSpacerItem>
#include <QCursor>
#include <QToolTip>
#include <QString>



SquirtleBotCommander::SquirtleBotCommander()
: rqt_gui_cpp::Plugin()

{
    //setObjectName("SquirtleBotCommander");
    //QMetaObject::connectSlotsByName(this);
}

SquirtleBotCommander::~SquirtleBotCommander() 
{
  m_publisher_exp.shutdown();
  m_publisher_ord.shutdown();
}

void SquirtleBotCommander::initPlugin(qt_gui_cpp::PluginContext& context)
{
    m_widget = new QWidget();
    m_ui.setupUi(m_widget);

    int serialNum = context.serialNumber();
    if( serialNum > 1) {
        m_widget->setWindowTitle(m_widget->windowTitle() + " (" + QString::number(serialNum) + ")");
    }
    context.addWidget(m_widget);
    
    connect(m_ui.explore_button, SIGNAL(clicked()), this, SLOT(on_exploreBut_clicked()));
    connect(m_ui.order_button, SIGNAL(clicked()), this, SLOT(on_orderBut_clicked()));
    
    m_publisher_exp = getNodeHandle().advertise<std_msgs::String>("/post/squirtle_fsm", 1000);
    m_publisher_ord = getNodeHandle().advertise<std_msgs::String>("/delivery_request", 1000);

}

void SquirtleBotCommander::shutdownPlugin()
{
}

void SquirtleBotCommander::saveSettings(qt_gui_cpp::Settings& plugin_settings, qt_gui_cpp::Settings& instance_settings) const
{
}

void SquirtleBotCommander::restoreSettings(const qt_gui_cpp::Settings& plugin_settings, const qt_gui_cpp::Settings& instance_settings)
{
}
  

void SquirtleBotCommander::on_exploreBut_clicked() {
  m_publisher_exp.shutdown();
  std_msgs::String msg;
  msg.data = "done_exploring";
  m_publisher_exp = getNodeHandle().advertise<std_msgs::String>("/post/squirtle_fsm", 1000);
  m_publisher_exp.publish(msg);
}

void SquirtleBotCommander::on_orderBut_clicked() {
  m_publisher_ord.shutdown();
  std_msgs::String msg;
  std::string text = m_ui.order_text->toPlainText().toStdString();;
  msg.data = text;
  m_publisher_ord = getNodeHandle().advertise<std_msgs::String>("/delivery_request", 1000);
  m_publisher_ord.publish(msg);
}


PLUGINLIB_EXPORT_CLASS(SquirtleBotCommander, rqt_gui_cpp::Plugin)
